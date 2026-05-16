import os, cv2, timm, torch, numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import gradio as gr

from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import densenet121

# ── Device ──────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
print(f"Device: {device}")

# ── ROI Extraction ───────────────────────────────────────────────
def extract_roi(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        pad = 20
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
        cropped = img[y1:y2, x1:x2]
        if cropped.shape[0] > 30 and cropped.shape[1] > 30:
            return cropped
    return img

# ── Transform ────────────────────────────────────────────────────
val_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

# ── Model ────────────────────────────────────────────────────────
class HybridModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=False)
        self.vit.reset_classifier(0)
        self.dense = densenet121(weights=None)
        self.dense.classifier = torch.nn.Identity()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(768 + 1024, 512),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(512),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 1),
        )

    def forward(self, x):
        v = self.vit(x)
        d = self.dense(x)
        return self.fc(torch.cat([v, d], dim=1))

model = HybridModel().to(device)
model.load_state_dict(torch.load("breast_model.pth", map_location=device))
model.eval()
print("Model loaded!")

# ── GradCAM++ ────────────────────────────────────────────────────
class GradCAMPlusPlus:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._fwd_hook)
        target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inp, out):
        self.activations = out.detach()

    def _bwd_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, img_tensor):
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(True)
        t = img_tensor.unsqueeze(0).to(device)
        t.requires_grad_(True)
        self.model.zero_grad()
        out = self.model(t)
        out[0, 0].backward()
        if self.gradients is None or self.activations is None:
            return np.zeros((7, 7))
        grads, acts = self.gradients, self.activations
        grads_sq = grads ** 2
        grads_cu = grads ** 3
        eps = 1e-8
        denom = 2.0 * grads_sq + (acts * grads_cu).sum(dim=[2, 3], keepdim=True) + eps
        alpha = grads_sq / denom
        weights = (alpha * torch.relu(grads)).sum(dim=[2, 3], keepdim=True)
        cam = torch.relu((weights * acts).sum(dim=1)).squeeze()
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

# ── Attention Rollout ────────────────────────────────────────────
class AttentionRollout:
    def __init__(self, model):
        self.model = model

    def get_attention_map(self, block, img_tensor):
        B, N, C = img_tensor.shape
        qkv = block.attn.qkv(img_tensor)
        qkv = qkv.reshape(B, N, 3, block.attn.num_heads, -1)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        scale = block.attn.scale
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        return attn

    def generate(self, img_tensor):
        self.model.eval()
        dev = next(self.model.parameters()).device
        t = img_tensor.unsqueeze(0).to(dev)
        with torch.no_grad():
            x = self.model.vit.patch_embed(t)
            cls_token = self.model.vit.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            x = self.model.vit.pos_drop(x + self.model.vit.pos_embed)
            all_attentions = []
            for block in self.model.vit.blocks:
                attn_norm = block.norm1(x)
                attn_map = self.get_attention_map(block, attn_norm)
                all_attentions.append(attn_map.squeeze(0).cpu())
                x = block(x)
        if not all_attentions:
            return np.zeros((14, 14))
        n_tokens = all_attentions[0].shape[-1]
        result = torch.eye(n_tokens)
        for attn in all_attentions:
            a = attn.mean(dim=0)
            a = a + torch.eye(n_tokens)
            a = a / a.sum(dim=-1, keepdim=True)
            result = torch.matmul(a, result)
        cls_attn = result[0, 1:]
        mask = cls_attn.reshape(14, 14).numpy()
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        return mask

# ── XAI Engines ──────────────────────────────────────────────────
gradcampp_engine = GradCAMPlusPlus(model, model.dense.features.denseblock4)
rollout_engine   = AttentionRollout(model)
print("XAI engines ready!")

# ── LIME ─────────────────────────────────────────────────────────
def run_lime(img_roi_rgb, num_samples=300):
    from lime import lime_image
    from skimage.segmentation import mark_boundaries

    img_224   = cv2.resize(img_roi_rgb, (224, 224))
    img_float = img_224.astype(np.float64) / 255.0

    def predict_fn(images):
        model.eval()
        tensors = []
        for img in images:
            img_uint8 = np.clip(img * 255, 0, 255).astype(np.uint8)
            tensors.append(val_tf(img_uint8))
        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            probs = torch.sigmoid(model(batch)).cpu().numpy().flatten()
        return np.column_stack([1.0 - probs, probs]).astype(np.float64)

    exp = lime_image.LimeImageExplainer(verbose=False)
    explanation = exp.explain_instance(
        img_float, predict_fn,
        top_labels=1, hide_color=0,
        num_samples=num_samples, batch_size=16,
        random_seed=42,
    )

    # available_labels() nahi hai is version mein — top_labels dict use karo
    label = list(explanation.top_labels)[0]
    print(f"[LIME] Using label: {label}")

    tp, mp = explanation.get_image_and_mask(label, positive_only=True,  num_features=8, hide_rest=False)
    ta, ma = explanation.get_image_and_mask(label, positive_only=False, num_features=8, hide_rest=False)

    return (mark_boundaries(tp, mp, color=(1, 0.2, 0.2)),
            mark_boundaries(ta, ma, color=(0.2, 0.8, 0.2)))

# ── Mammogram Validation ─────────────────────────────────────────
def is_valid_mammogram(img_rgb):
    h, w = img_rgb.shape[:2]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    if h < 64 or w < 64:
        return False, "Image bahut chhoti hai (minimum 64x64 pixels chahiye)."

    r = img_rgb[:,:,0].astype(int)
    g = img_rgb[:,:,1].astype(int)
    b = img_rgb[:,:,2].astype(int)
    channel_diff = max(
        np.mean(np.abs(r - g)),
        np.mean(np.abs(g - b)),
        np.mean(np.abs(r - b))
    )
    if channel_diff > 40:
        return False, "Colored image detect hui. Mammogram ek grayscale X-ray hoti hai."

    mean_brightness = np.mean(gray)
    if mean_brightness < 2:
        return False, "Image bilkul black hai. Valid mammogram upload karo."
    if mean_brightness > 250:
        return False, "Image bilkul white hai. Valid mammogram upload karo."

    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < 10:
        return False, "Image mein koi structure/texture nahi hai. Valid mammogram upload karo."

    return True, "OK"

# ── Main Explain Function ────────────────────────────────────────
def explain_image(img_input, lime_samples=300, alpha=0.5, threshold=0.5):
    try:
        # Image prep
        if isinstance(img_input, Image.Image):
            img_rgb = np.array(img_input.convert("RGB"))
        else:
            img_rgb = np.array(img_input)

        if img_rgb is None or img_rgb.size == 0:
            raise ValueError("Image empty hai — koi image upload karo.")

        # Validation
        valid, reason = is_valid_mammogram(img_rgb)
        if not valid:
            fig, ax = plt.subplots(figsize=(8, 4), facecolor="#0d0d0d")
            ax.set_facecolor("#0d0d0d")
            ax.text(0.5, 0.65, "⚠️  Invalid Image Detected",
                    ha="center", va="center", color="#EF9F27",
                    fontsize=14, fontweight="bold", transform=ax.transAxes)
            ax.text(0.5, 0.40, reason,
                    ha="center", va="center", color="#aaaaaa",
                    fontsize=11, transform=ax.transAxes, wrap=True)
            ax.text(0.5, 0.18, "Please upload a valid grayscale mammogram X-ray image.",
                    ha="center", va="center", color="#666666",
                    fontsize=9, fontstyle="italic", transform=ax.transAxes)
            ax.axis("off")
            return fig

        # ROI + Tensor
        img_roi    = extract_roi(img_rgb)
        img_224    = cv2.resize(img_roi, (224, 224))
        img_tensor = val_tf(img_roi)

        # Prediction
        model.eval()
        with torch.no_grad():
            prob = torch.sigmoid(model(img_tensor.unsqueeze(0).to(device))).item()

        label = "MALIGNANT" if prob > threshold else "BENIGN"
        color = "#E8593C"   if prob > threshold else "#5DCAA5"

        # GradCAM++
        cam     = gradcampp_engine.generate(img_tensor)
        cam_up  = cv2.resize(cam, (224, 224))
        gc_heat = (cm.jet(cam_up)[:, :, :3] * 255).astype(np.uint8)
        gc_over = (alpha * gc_heat + (1 - alpha) * img_224).astype(np.uint8)

        # Attention Rollout
        attn    = rollout_engine.generate(img_tensor)
        attn_up = cv2.resize(attn, (224, 224))
        at_heat = (cm.viridis(attn_up)[:, :, :3] * 255).astype(np.uint8)
        at_over = (alpha * at_heat + (1 - alpha) * img_224).astype(np.uint8)

        # LIME — mandatory, hamesha chalega
        print("[LIME] Starting...")
        lime_pos, lime_all = run_lime(img_roi, int(lime_samples))
        print("[LIME] Done!")

        # Plot — hamesha 4 columns
        fig = plt.figure(figsize=(24, 10), facecolor="#0d0d0d")
        fig.text(
            0.5, 0.97,
            f"Prediction: {label}   |   P(malignant) = {prob:.3f}   |   P(benign) = {1-prob:.3f}   |   Threshold = {threshold}",
            ha="center", va="top", fontsize=16, color=color, fontweight="bold",
        )
        gs = gridspec.GridSpec(
            2, 4, figure=fig,
            hspace=0.30, wspace=0.06,
            top=0.92, bottom=0.04,
            left=0.04, right=0.97,
        )

        def _ax(row, col, img, title, cmap=None, colorbar=False):
            ax = fig.add_subplot(gs[row, col])
            ax.set_facecolor("#1a1a1a")
            im = ax.imshow(img, cmap=cmap)
            ax.set_title(title, color="white", fontsize=11, pad=5)
            ax.axis("off")
            if colorbar:
                cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
                plt.setp(cb.ax.yaxis.get_ticklabels(), color="white", fontsize=7)

        _ax(0, 0, img_224,  "Original ROI",                          "gray")
        _ax(0, 1, cam_up,   "GradCAM++ Heatmap",                     "jet",     colorbar=True)
        _ax(0, 2, gc_over,  "GradCAM++ Overlay")
        _ax(0, 3, lime_pos, "LIME — Positive Regions\n(Red = Malignancy Support)")

        _ax(1, 0, img_224,  "Original ROI",                          "gray")
        _ax(1, 1, attn_up,  "ViT Attention Rollout",                 "viridis", colorbar=True)
        _ax(1, 2, at_over,  "Attention Overlay")
        _ax(1, 3, lime_all, "LIME — All Regions\n(Red=Malignant · Green=Benign)")

        for row, txt in [(0, "DenseNet  /  GradCAM++"), (1, "ViT  /  Attention Rollout")]:
            ax = fig.add_subplot(gs[row, :])
            ax.set_axis_off()
            ax.text(-0.008, 0.5, txt,
                    transform=ax.transAxes,
                    fontsize=9, color="#666", va="center",
                    rotation=90, ha="right")

        return fig

    except Exception as e:
        import traceback
        traceback.print_exc()
        fig, ax = plt.subplots(facecolor="#0d0d0d")
        ax.set_facecolor("#0d0d0d")
        ax.text(0.5, 0.5, f"Error:\n{str(e)}",
                ha="center", va="center", color="#E8593C",
                fontsize=12, transform=ax.transAxes, wrap=True)
        ax.axis("off")
        return fig

# ── Gradio UI ────────────────────────────────────────────────────
with gr.Blocks(
    theme=gr.themes.Base(primary_hue="orange", neutral_hue="zinc"),
    title="Mammography XAI",
    css="""
    .gr-button-primary { background: #E8593C !important; border: none !important; }
    footer { display: none !important; }
    """
) as demo:

    gr.Markdown("""
    #XAI Dashboard
    Upload a mammogram image 
    """)

    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(
                label="Upload Mammogram Image",
                type="pil",
                height=300,
            )
            with gr.Accordion("⚙️ Settings", open=False):
                lime_slider = gr.Slider(
                    minimum=100, maximum=1000, value=300, step=50,
                    label="LIME samples (zyada = slow but accurate)",
                )
                alpha_slider = gr.Slider(
                    minimum=0.1, maximum=0.9, value=0.5, step=0.05,
                    label="Overlay opacity",
                )
                thresh_slider = gr.Slider(
                    minimum=0.1, maximum=0.9, value=0.5, step=0.05,
                    label="Malignancy threshold",
                )
            run_btn = gr.Button("🔍 Analyze", variant="primary", size="lg")

        with gr.Column(scale=3):
            plot_out = gr.Plot(label="XAI Dashboard")

    run_btn.click(
        fn=explain_image,
        inputs=[img_input, lime_slider, alpha_slider, thresh_slider],
        outputs=[plot_out],
    )

    gr.Markdown("""
    ---
    **GradCAM++** — DenseNet denseblock4 par second-order gradient attention.
    **Attention Rollout** — ViT ke sabhi 12 layers ka cumulative patch importance.
    **LIME** — Red = malignancy support · Green = benign support
    *Model: ViT-B/16 + DenseNet-121 | Dataset: CBIS-DDSM*
    """)

demo.launch()





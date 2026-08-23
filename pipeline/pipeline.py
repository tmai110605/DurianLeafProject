import os
import sys

# Add project root to sys.path to enable imports of models and pipeline
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Force UTF-8 encoding for Windows console to support Vietnamese characters print
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

import cv2
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import pipeline as hf_pipeline
# pyrefly: ignore [missing-import]
import plotly.graph_objects as go
from models.mobilenetv3_custom import MobileNetV2MultiTask
from pipeline.recommender import *
from pipeline.weather_kb import *
import time
import thop

_DEPTH_ANYTHING_FLOPS = None
_DEPTH_ANYTHING_PARAMS = None

def get_depth_anything_flops_and_params(model, device="cpu"):
    global _DEPTH_ANYTHING_FLOPS, _DEPTH_ANYTHING_PARAMS
    if _DEPTH_ANYTHING_FLOPS is None:
        try:
            import torch
            dummy_input = torch.randn(1, 3, 518, 518).to(device)
            # Run thop profile silently
            flops, params = thop.profile(model, inputs=(dummy_input,), verbose=False)
            _DEPTH_ANYTHING_FLOPS = flops
            _DEPTH_ANYTHING_PARAMS = params
        except Exception as e:
            print(f"Error profiling Depth-Anything V2 with thop: {e}")
            _DEPTH_ANYTHING_FLOPS = 0.0
            _DEPTH_ANYTHING_PARAMS = 0.0
    return _DEPTH_ANYTHING_FLOPS, _DEPTH_ANYTHING_PARAMS

# =====================================================
# DEVICE
# =====================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# KNOWLEDGE BASE (load 1 lần khi import)
# =====================================================

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
KB_PATH = os.path.join(_CURRENT_DIR, "knowledge_base", "durian_leaf_case_based_recommendation_kb.json")

KB = DurianLeafKB(KB_PATH)

# =====================================================
# LABELS
# =====================================================

DISEASE_LABELS = [
    "class_0",
    "class_1",
    "class_2",
    "class_3",
    "class_4",
]

SEVERITY_LABELS = [
    "severity_0",
    "severity_1",
    "severity_2",
    "severity_3",
]

# =====================================================
# TRANSFORMS
# =====================================================

def build_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    ])


def load_image(image_path, img_size=224):
    pil_img = Image.open(image_path).convert("RGB")
    transform = build_transform(img_size)
    input_tensor = transform(pil_img).unsqueeze(0)
    original_rgb = np.array(pil_img.resize((img_size, img_size)))
    return input_tensor, original_rgb

# =====================================================
# CHECKPOINT
# =====================================================

def clean_state_dict(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        new_state_dict[k] = v
    return new_state_dict


def load_checkpoint(model, ckpt_path, device=DEVICE):
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    state_dict = clean_state_dict(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("Missing keys   :", missing)
    print("Unexpected keys:", unexpected)

    model.to(device)
    model.eval()
    return model

# =====================================================
# GRAD-CAM
# =====================================================

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.forward_handle = target_layer.register_forward_hook(self._save_activation)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def __call__(self, input_tensor, task="disease", class_idx=None):
        self.model.zero_grad()

        disease_logits, severity_logits = self.model(input_tensor)
        logits = disease_logits if task == "disease" else severity_logits

        probs = F.softmax(logits, dim=1)

        if class_idx is None:
            class_idx = torch.argmax(probs, dim=1).item()

        logits[:, class_idx].backward(retain_graph=True)

        weights = self.gradients[0].mean(dim=(1, 2), keepdim=True)
        cam = torch.sum(weights * self.activations[0], dim=0)
        cam = F.relu(cam).cpu().numpy()

        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        return cam, class_idx, probs.detach().cpu().numpy()[0]

# =====================================================
# OVERLAY 2D
# =====================================================

def overlay_cam_on_image(rgb_img, cam, alpha=0.45):
    h, w, _ = rgb_img.shape
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.clip(heatmap * alpha + rgb_img * (1 - alpha), 0, 255).astype(np.uint8)
    return overlay

# =====================================================
# DEPTH ESTIMATION
# =====================================================

def estimate_depth(pil_img, max_size=200):
    """
    Trả về depth_small đã normalize và scale,
    cùng với (new_h, new_w) để resize CAM theo, và một dict chứa các thông số đo đạc.
    """
    t_load_start = time.time()
    depth_pipe = hf_pipeline(
        "depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf"
    )
    t_load_end = time.time()

    t_infer_start = time.time()
    depth = np.array(
        depth_pipe(pil_img)["depth"]
    ).astype(np.float32)
    t_infer_end = time.time()

    # Đo FLOPs và Params bằng thop
    flops, params = get_depth_anything_flops_and_params(depth_pipe.model, device=DEVICE)

    h, w = depth.shape
    scale = min(max_size / h, max_size / w)
    new_h, new_w = int(h * scale), int(w * scale)

    depth_small = np.array(
        Image.fromarray(depth).resize((new_w, new_h), Image.BILINEAR)
    )

    depth_small = (depth_small - depth_small.min()) / (depth_small.max() - depth_small.min())
    depth_small = depth_small * 50

    metrics = {
        "load_time": t_load_end - t_load_start,
        "infer_time": t_infer_end - t_infer_start,
        "flops": flops,
        "params": params,
    }

    return depth_small, new_h, new_w, metrics

# =====================================================
# 3D VISUALIZATION (Geo-GradCAM)
# =====================================================

def visualize_geo_gradcam(
    depth_small,
    cam_resized,
    new_h,
    new_w,
    task_name="severity",
    attention_boost=0.4,
    output_html=None,
    show=False,
):
    """
    Vẽ Geo-GradCAM: depth surface modulated bởi CAM attention.

    Parameters
    ----------
    depth_small     : np.ndarray (new_h, new_w) — depth đã normalize
    cam_resized     : np.ndarray (new_h, new_w) — CAM đã resize & normalize
    task_name       : str — tên task để hiển thị trên title
    attention_boost : float — hệ số boost vùng attention lên trục Z
    output_html     : str hoặc None — nếu có, lưu file html
    show            : bool — hiển thị trực tiếp (dùng khi chạy script)
    """
    z_visual = depth_small * (1 + attention_boost * cam_resized)

    x = np.arange(new_w)
    y = np.arange(new_h - 1, -1, -1)   # flip Y để khớp hệ tọa độ ảnh

    fig = go.Figure(
        data=[
            go.Surface(
                z=z_visual,
                x=x,
                y=y,
                surfacecolor=cam_resized,
                colorscale="Jet",
                cmin=0,
                cmax=1,
                showscale=True,
                colorbar=dict(title="Grad-CAM")
            )
        ]
    )

    fig.update_layout(
        title=f"Geo-GradCAM — {task_name}",
        width=1000,
        height=800,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Depth"
        )
    )

    if output_html:
        fig.write_html(output_html)
        print(f"Saved 3D plot -> {output_html}")

    if show:
        fig.show()

    return fig

# =====================================================
# MAIN PIPELINE
# =====================================================

def run_pipeline(
    image_path,
    checkpoint_path,
    tasks=("disease", "severity"),
    img_size=224,
    max_size=200,
    attention_boost=0.4,
    output_dir="output",
    save_2d_overlay=True,
    save_3d_html=True,
    show_3d=False,
    lat=None,
    lon=None,
    use_weather=True,
):
    """
    Pipeline đầy đủ:
      0. Lấy vị trí + thời tiết (nếu use_weather=True)
      1. Load ảnh
      2. Depth estimation (Depth Anything V2)
      3. Load model + phân loại bệnh & severity
      4. Grad-CAM (cho mỗi task trong tasks)
      5. Align CAM với depth map + vẽ Geo-GradCAM 3D
      +  LLM recommendation

    Returns
    -------
    dict chứa prediction results, recommendation, và paths file đã lưu.
    """
    execution_times = {}
    total_start = time.time()
    
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(str(image_path)))[0]

    # ── 0. Vị trí + thời tiết ──────────────────────
    t_weather_start = time.time()
    weather_data = None
    location     = None

    if use_weather:
        print("[0/5] Getting location & weather...")
        try:
            if lat is None or lon is None:
                location = get_location_from_ip()
                lat, lon = location["lat"], location["lon"]
            else:
                location = get_address_from_coords(lat, lon)

            weather_data = get_weather_forecast(lat, lon)
        except Exception as e:
            print(f"  Weather lookup failed: {e}")
            weather_data = None
    execution_times["weather_location"] = time.time() - t_weather_start

    # ── 1. Load ảnh ────────────────────────────────
    t_img_start = time.time()
    print("[1/5] Loading image...")
    pil_img = Image.open(image_path).convert("RGB")
    execution_times["image_loading"] = time.time() - t_img_start

    # ── 2. Depth Estimation ────────────────────────
    print("[2/5] Running Depth Anything V2...")
    t_depth_start = time.time()
    depth_small, new_h, new_w, depth_metrics = estimate_depth(pil_img, max_size=max_size)
    execution_times["depth_model_load"] = depth_metrics["load_time"]
    execution_times["depth_inference"] = depth_metrics["infer_time"]
    execution_times["depth_estimation"] = time.time() - t_depth_start
    depth_flops = depth_metrics["flops"]
    depth_params = depth_metrics["params"]

    # ── 3. Load model + phân loại ──────────────────
    print("[3/5] Loading model & computing predictions...")
    t_classifier_load_start = time.time()
    model = MobileNetV2MultiTask(
        num_disease_classes=5,
        num_severity_classes=4
    )
    model = load_checkpoint(model, checkpoint_path, DEVICE)
    execution_times["classifier_model_loading"] = time.time() - t_classifier_load_start

    t_classifier_infer_start = time.time()
    input_tensor, original_rgb = load_image(image_path, img_size)
    input_tensor = input_tensor.to(DEVICE)

    with torch.no_grad():
        disease_logits, severity_logits = model(input_tensor)
        disease_probs  = F.softmax(disease_logits,  dim=1)[0]
        severity_probs = F.softmax(severity_logits, dim=1)[0]
        disease_idx    = torch.argmax(disease_probs).item()
        severity_idx   = torch.argmax(severity_probs).item()
    execution_times["classifier_inference"] = time.time() - t_classifier_infer_start

    print(f"  Disease  -> {DISEASE_LABELS[disease_idx]}  ({disease_probs[disease_idx]:.4f})")
    print(f"  Severity -> {SEVERITY_LABELS[severity_idx]} ({severity_probs[severity_idx]:.4f})")

    # ── Weather scenario ───────────────────────────
    weather_scenario = "WS_NORMAL"
    if weather_data is not None:
        weather_scenario = determine_weather_scenario(KB.weather_scenarios, weather_data)
        print(f"  Weather scenario: {weather_scenario}")

    # ── LLM Recommendation ─────────────────────────
    recommendation = get_recommendation(
        disease_idx=disease_idx,
        severity_idx=severity_idx,
        disease_confidence=disease_probs[disease_idx].item(),
        severity_confidence=severity_probs[severity_idx].item(),
        kb=KB,
        model="openai/gpt-oss-120b",
        weather_scenario=weather_scenario,
        weather_data=weather_data,
        location=location,
    )
    execution_times["groq_call"] = recommendation.get("groq_time", 0.0)

    print("\n===== RECOMMENDATION =====\n")
    if recommendation["error"]:
        print(recommendation["error"])
    else:
        print(recommendation["recommendation_text"])

    # ── 4. Grad-CAM + 5. Geo-GradCAM ──────────────
    print("[4/5] Computing Grad-CAM...")
    t_gradcam_start = time.time()
    gradcam = GradCAM(model, model.features[-1])

    task_results = {}
    t_gradcam_calc_total = 0.0
    t_rendering_total = 0.0
    
    for task_for_cam in tasks:
        print(f"  [{task_for_cam}] computing GradCAM...")

        t_calc_start = time.time()
        cam, cam_class_idx, cam_probs = gradcam(
            input_tensor, task=task_for_cam, class_idx=None
        )
        t_gradcam_calc_total += (time.time() - t_calc_start)

        # ── Resize CAM → ảnh gốc (cho 2D overlay) ─
        cam_2d = cv2.resize(cam, (img_size, img_size))
        cam_2d = (cam_2d - cam_2d.min()) / (cam_2d.max() - cam_2d.min() + 1e-8)

        # ── Resize CAM → depth size (cho 3D) ───────
        cam_3d = cv2.resize(cam, (new_w, new_h))
        cam_3d = (cam_3d - cam_3d.min()) / (cam_3d.max() - cam_3d.min() + 1e-8)

        task_data = {
            "gradcam_class_idx": cam_class_idx,
            "overlay_2d_path":   None,
            "geo_gradcam_html":  None,
        }

        t_render_start = time.time()
        # ── 4a. Lưu 2D overlay ────────────────────
        if save_2d_overlay:
            overlay = overlay_cam_on_image(original_rgb, cam_2d)
            overlay_path = os.path.join(
                output_dir, f"{base_name}_gradcam2d_{task_for_cam}.jpg"
            )
            Image.fromarray(overlay).save(overlay_path)
            print(f"  Saved 2D overlay -> {overlay_path}")
            task_data["overlay_2d_path"] = overlay_path

        # ── 4b. Geo-GradCAM 3D HTML ───────────────
        if save_3d_html:
            print(f"[5/5] [{task_for_cam}] Rendering Geo-GradCAM 3D...")
            html_path = os.path.join(
                output_dir, f"{base_name}_geo_gradcam_{task_for_cam}.html"
            )
            visualize_geo_gradcam(
                depth_small=depth_small,
                cam_resized=cam_3d,
                new_h=new_h,
                new_w=new_w,
                task_name=task_for_cam,
                attention_boost=attention_boost,
                output_html=html_path,
                show=show_3d,
            )
            task_data["geo_gradcam_html"] = html_path
        t_rendering_total += (time.time() - t_render_start)

        task_results[task_for_cam] = task_data

    gradcam.remove_hooks()
    
    execution_times["gradcam_computation"] = t_gradcam_calc_total
    execution_times["geo_gradcam_rendering"] = t_rendering_total
    execution_times["total_pipeline"] = time.time() - total_start

    return {
        "disease_idx":          disease_idx,
        "disease_label":        DISEASE_LABELS[disease_idx],
        "disease_confidence":   disease_probs[disease_idx].item(),
        "severity_idx":         severity_idx,
        "severity_label":       SEVERITY_LABELS[severity_idx],
        "severity_confidence":  severity_probs[severity_idx].item(),
        "weather_scenario":     weather_scenario,
        "weather":              weather_data,
        "location":             location,
        "recommendation":       recommendation,
        "tasks":                task_results,
        "execution_times":      execution_times,
        "depth_flops":          depth_flops,
        "depth_params":         depth_params,
    }


# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":
    # Đảm bảo thư mục làm việc hiện tại luôn là thư mục gốc của dự án
    _APP_DIR = os.path.dirname(os.path.abspath(__file__))
    _PROJECT_ROOT = os.path.dirname(_APP_DIR)
    os.chdir(_PROJECT_ROOT)

    IMAGE_PATH      = r"data\Durian_Leaf_Diseases\test\algal\DLDD_TEST_000006.jpg"
    CHECKPOINT_PATH = r"checkpoints\v2batch16_best\best_mobilenetv2_multitask.pth"

    result = run_pipeline(
        image_path=IMAGE_PATH,
        checkpoint_path=CHECKPOINT_PATH,
        tasks=("disease", "severity"),
        img_size=224,
        max_size=200,
        attention_boost=0.4,
        output_dir="output",
        save_2d_overlay=True,
        save_3d_html=True,
        show_3d=False,
    )

    print("\n-- Results --")
    print(f"  disease_label  : {result['disease_label']} ({result['disease_confidence']:.4f})")
    print(f"  severity_label : {result['severity_label']} ({result['severity_confidence']:.4f})")
    for task_name, task_data in result["tasks"].items():
        print(f"  [{task_name}]")
        for k, v in task_data.items():
            print(f"      {k}: {v}")
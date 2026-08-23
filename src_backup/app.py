"""
Durian Leaf Disease Detection — Streamlit App
Wraps the existing pipeline.py into an interactive web interface.
"""

import os
import sys
import tempfile
import streamlit as st
from PIL import Image
import numpy as np
from streamlit_geolocation import streamlit_geolocation

# ── Fix working directory ─────────────────────────────────────────────────────
# app/app.py nằm trong app/, project root là thư mục cha.
_APP_DIR      = os.path.dirname(os.path.abspath(__file__))  # .../app
_PROJECT_ROOT = os.path.dirname(_APP_DIR)                   # project root
os.chdir(_PROJECT_ROOT)

# Thêm project root vào sys.path để import các package: pipeline, models, training
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ── Label mappings (giữ đồng bộ với recommender.py) ─────────────────────────
DISEASE_IDX_TO_LABEL = {
    0: "healthy",
    1: "algal",
    2: "allocaridara_attack",
    3: "blight",
    4: "phomopsis",
}
SEVERITY_IDX_TO_LEVEL = {0: 0, 1: 1, 2: 2, 3: 3}

DISEASE_DISPLAY = {
    "healthy":              "🟢 Healthy",
    "algal":                "🔵 Algal Spot",
    "allocaridara_attack":  "🟠 Allocaridara Attack",
    "blight":               "🔴 Blight",
    "phomopsis":            "🟣 Phomopsis",
}
SEVERITY_DISPLAY = {
    0: "severity_0 — Healthy",
    1: "severity_1 — Mild",
    2: "severity_2 — Moderate",
    3: "severity_3 — Severe",
}
SEVERITY_COLORS = ["badge-0", "badge-1", "badge-2", "badge-3"]


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Durian Leaf Diagnostics",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Sora:wght@300;400;600;700&display=swap');

/* ── Root palette ── */
:root {
    --bg:        #0d1208;
    --surface:   #131a0e;
    --border:    #2a3820;
    --accent:    #7ec850;
    --accent-dim:#3d6626;
    --warn:      #e8a838;
    --danger:    #e05252;
    --text:      #d6e8c4;
    --muted:     #6e8c5a;
    --mono:      'DM Mono', monospace;
    --sans:      'Sora', sans-serif;
}

/* ── Global resets ── */
html, body, [class*="css"]  { font-family: var(--sans); }
.stApp                      { background: var(--bg); color: var(--text); }
section[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border); }

/* ── Headings ── */
h1 { font-size: 2rem; font-weight: 700; letter-spacing: -0.03em; color: var(--accent); margin-bottom: 0; }
h2 { font-size: 1.1rem; font-weight: 600; color: var(--text); }
h3 { font-size: 0.9rem; font-weight: 500; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; }

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    border: 1.5px dashed var(--accent-dim);
    border-radius: 8px;
    padding: 1rem;
    background: rgba(126,200,80,0.04);
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent); }

/* ── Metric cards ── */
.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.1rem 1.4rem;
    display: flex;
    flex-direction: column;
    gap: 4px;
}
.metric-label { font-family: var(--mono); font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; }
.metric-value { font-family: var(--mono); font-size: 1.35rem; font-weight: 500; color: var(--accent); }
.metric-sub   { font-size: 0.78rem; color: var(--muted); }

/* ── Confidence bar ── */
.conf-bar-wrap { background: var(--border); border-radius: 4px; height: 6px; width: 100%; overflow: hidden; margin-top: 6px; }
.conf-bar      { height: 100%; border-radius: 4px; background: linear-gradient(90deg, var(--accent-dim), var(--accent)); transition: width 0.5s; }

/* ── Recommendation box ── */
.rec-box {
    background: linear-gradient(135deg, rgba(62,100,38,0.18), rgba(13,18,8,0.9));
    border: 1px solid var(--accent-dim);
    border-left: 3px solid var(--accent);
    border-radius: 10px;
    padding: 1.3rem 1.5rem;
    margin-top: 0.5rem;
    font-size: 0.9rem;
    line-height: 1.7;
    color: var(--text);
    white-space: pre-wrap;
}

/* ── Weather card ── */
.weather-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--warn);
    border-radius: 10px;
    padding: 1rem 1.4rem;
    font-size: 0.85rem;
    line-height: 1.6;
    color: var(--text);
}
.weather-card .wlabel { font-family: var(--mono); font-size: 0.7rem; color: var(--warn); text-transform: uppercase; letter-spacing: 0.1em; }

/* ── Source list ── */
.source-list {
    background: rgba(126,200,80,0.04);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    font-size: 0.8rem;
    color: var(--muted);
    line-height: 1.8;
}

/* ── Severity badge ── */
.badge {
    display: inline-block;
    font-family: var(--mono);
    font-size: 0.75rem;
    padding: 2px 10px;
    border-radius: 20px;
    font-weight: 500;
}
.badge-0 { background: rgba(126,200,80,0.15); color: #7ec850; border: 1px solid #3d6626; }
.badge-1 { background: rgba(232,168,56,0.15); color: #e8a838; border: 1px solid #7a5210; }
.badge-2 { background: rgba(224,82,82,0.13);  color: #e05252; border: 1px solid #7a2020; }
.badge-3 { background: rgba(180,50,50,0.18);  color: #e05252; border: 1px solid #8b1a1a; }

/* ── Section divider ── */
.divider { border: none; border-top: 1px solid var(--border); margin: 1.5rem 0; }

/* ── Streamlit overrides ── */
.stButton > button {
    background: var(--accent);
    color: var(--bg);
    font-family: var(--sans);
    font-weight: 600;
    border: none;
    border-radius: 7px;
    padding: 0.55rem 1.4rem;
    font-size: 0.9rem;
    letter-spacing: 0.02em;
    transition: opacity 0.15s;
}
.stButton > button:hover { opacity: 0.85; }

[data-testid="stSelectbox"] label,
[data-testid="stSlider"]    label { font-size: 0.82rem; color: var(--muted); }

/* Sidebar labels */
.sidebar-label {
    font-family: var(--mono);
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted);
    margin-bottom: 2px;
}

/* Tabs */
[data-testid="stTab"] { color: var(--muted); }
[data-testid="stTab"][aria-selected="true"] { color: var(--accent) !important; border-bottom: 2px solid var(--accent); }

/* Info / warning boxes */
.stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 Durian Leaf\n### Diagnostics System")
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    st.markdown('<p class="sidebar-label">Model checkpoint</p>', unsafe_allow_html=True)
    checkpoint_path = st.text_input(
        "Checkpoint path",
        value=r"checkpoints/v2batch16_best/best_mobilenetv2_multitask.pth",
        label_visibility="collapsed",
        placeholder="path/to/checkpoint.pth",
    )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-label">Inference settings</p>', unsafe_allow_html=True)

    img_size = st.select_slider("Input resolution", options=[128, 160, 192, 224, 256], value=224)
    max_size = st.slider("Depth map resolution", 100, 300, 200, step=20)
    attention_boost = st.slider("GradCAM attention boost", 0.0, 1.0, 0.4, step=0.05)

    tasks_selected = st.multiselect(
        "GradCAM tasks",
        options=["disease", "severity"],
        default=["disease", "severity"],
    )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-label">Output options</p>', unsafe_allow_html=True)
    save_2d = st.checkbox("Save 2D overlay", value=True)
    save_3d = st.checkbox("Save 3D Geo-GradCAM HTML", value=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    use_weather = st.checkbox("Lấy thông tin thời tiết tự động", value=True)

    gps_lat, gps_lon = None, None
    if use_weather:
        st.markdown('<p class="sidebar-label">📍 Định vị GPS (trình duyệt)</p>', unsafe_allow_html=True)
        location_data = streamlit_geolocation()
        if location_data and location_data.get("latitude") is not None:
            gps_lat = location_data.get("latitude")
            gps_lon = location_data.get("longitude")
            st.markdown(
                f'<p style="color:var(--accent);font-size:0.8rem;margin-top:2px;font-family:var(--mono);">'
                f'Đã định vị: {gps_lat:.4f}, {gps_lon:.4f}'
                f'</p>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<p style="color:var(--muted);font-size:0.75rem;margin-top:2px;">'
                'Click vào nút bên trên để lấy tọa độ thực tế. Nếu không, hệ thống sẽ fallback sang định vị địa chỉ IP.'
                '</p>',
                unsafe_allow_html=True
            )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.caption("MobileNetV2 · Depth-Anything V2 · Grad-CAM · Groq LLM")


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("<h1>Durian Leaf Disease Detection</h1>", unsafe_allow_html=True)
st.markdown(
    '<p style="color:var(--muted);font-size:0.88rem;margin-top:2px;margin-bottom:1.5rem;">'
    'Multi-task classification · Geo-GradCAM 3D visualisation · LLM treatment recommendation'
    '</p>',
    unsafe_allow_html=True,
)

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Drop a durian leaf image here, or click to browse",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="visible",
)

if uploaded_file is None:
    # Landing state
    st.markdown("""
    <div style="margin-top:3rem;text-align:center;color:var(--muted);">
        <div style="font-size:3rem;">🍃</div>
        <p style="margin-top:0.5rem;font-size:0.9rem;">
            Upload a leaf image to start diagnosis.<br>
            Adjust model settings in the sidebar.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Validate inputs ───────────────────────────────────────────────────────────
if not checkpoint_path.strip():
    st.error("⚠️ Please enter the model checkpoint path in the sidebar.")
    st.stop()

if not tasks_selected:
    st.warning("Select at least one GradCAM task in the sidebar.")
    st.stop()


# ── Run pipeline on button press ──────────────────────────────────────────────
col_img, col_run = st.columns([3, 1])

with col_img:
    preview = Image.open(uploaded_file)
    st.image(preview, caption=uploaded_file.name, use_container_width=True)

with col_run:
    st.markdown("<br><br>", unsafe_allow_html=True)
    run_btn = st.button("▶ Run Diagnosis", use_container_width=True)

if not run_btn:
    st.stop()


# ── Execute ───────────────────────────────────────────────────────────────────
progress = st.progress(0, text="Saving image...")

# Write uploaded file to a temp path so pipeline can read it
with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
    tmp.write(uploaded_file.getvalue())
    tmp_image_path = tmp.name

output_dir = tempfile.mkdtemp(prefix="durian_output_")

try:
    # Lazy import — user must have pipeline.py on PYTHONPATH
    from pipeline.pipeline import run_pipeline  # noqa: E402

    progress.progress(5, text="[1/5] Loading image…")
    progress.progress(10, text="[2/5] Running Depth-Anything V2…")

    result = run_pipeline(
        image_path=tmp_image_path,
        checkpoint_path=checkpoint_path,
        tasks=tuple(tasks_selected),
        img_size=img_size,
        max_size=max_size,
        attention_boost=attention_boost,
        output_dir=output_dir,
        save_2d_overlay=save_2d,
        save_3d_html=save_3d,
        show_3d=False,
        lat=gps_lat,
        lon=gps_lon,
        use_weather=use_weather,
    )

    progress.progress(100, text="✅ Done!")

except ImportError:
    progress.empty()
    st.error(
        "**`pipeline.py` not found.** "
        "Make sure `pipeline.py` (and its dependencies) are on your Python path "
        "and that you're running `streamlit run app.py` from the project root."
    )
    st.stop()
except FileNotFoundError as exc:
    progress.empty()
    st.error(f"File not found: {exc}")
    st.stop()
except Exception as exc:
    progress.empty()
    st.exception(exc)
    st.stop()
finally:
    try:
        os.unlink(tmp_image_path)
    except OSError:
        pass


# ── Helper ────────────────────────────────────────────────────────────────────
def conf_bar(conf: float) -> str:
    pct = int(conf * 100)
    return (
        f'<div class="conf-bar-wrap"><div class="conf-bar" style="width:{pct}%"></div></div>'
        f'<span class="metric-sub">{pct}% confidence</span>'
    )


# ── Results ───────────────────────────────────────────────────────────────────
d_idx   = result["disease_idx"]
d_conf  = result["disease_confidence"]
sv_idx  = result["severity_idx"]
sv_conf = result["severity_confidence"]

# Dùng đúng label mapping từ recommender.py
d_label_key  = DISEASE_IDX_TO_LABEL.get(d_idx, f"class_{d_idx}")
d_label_disp = DISEASE_DISPLAY.get(d_label_key, d_label_key)
sv_label_disp = SEVERITY_DISPLAY.get(sv_idx, f"severity_{sv_idx}")
badge_cls     = SEVERITY_COLORS[sv_idx] if sv_idx < len(SEVERITY_COLORS) else "badge-0"

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("## Diagnosis Results")

# ── Metric cards row ──────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(f"""
    <div class="metric-card">
        <span class="metric-label">Disease class</span>
        <span class="metric-value">{d_label_disp}</span>
        {conf_bar(d_conf)}
    </div>""", unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-card">
        <span class="metric-label">Disease confidence</span>
        <span class="metric-value">{d_conf:.1%}</span>
        <span class="metric-sub">{d_conf:.4f} logit score</span>
    </div>""", unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-card">
        <span class="metric-label">Severity level</span>
        <span class="metric-value">{sv_label_disp}</span>
        <span class="badge {badge_cls}">{sv_idx} / 3</span>
    </div>""", unsafe_allow_html=True)
with m4:
    st.markdown(f"""
    <div class="metric-card">
        <span class="metric-label">Severity confidence</span>
        <span class="metric-value">{sv_conf:.1%}</span>
        {conf_bar(sv_conf)}
    </div>""", unsafe_allow_html=True)


# ── Latency & Performance Breakdown ───────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("## ⚡ Performance & Latency Breakdown")

exec_times = result.get("execution_times", {})
t_img_load = exec_times.get("image_loading", 0.0)
t_cls_load = exec_times.get("classifier_model_loading", 0.0)
t_cls_infer = exec_times.get("classifier_inference", 0.0)
t_diag = t_img_load + t_cls_load + t_cls_infer

t_depth_load = exec_times.get("depth_model_load", 0.0)
t_depth_infer = exec_times.get("depth_inference", 0.0)
t_cam_calc = exec_times.get("gradcam_computation", 0.0)
t_render = exec_times.get("geo_gradcam_rendering", 0.0)
t_gradcam = t_depth_load + t_depth_infer + t_cam_calc + t_render

t_groq = exec_times.get("groq_call", 0.0)
t_total = exec_times.get("total_pipeline", 0.0)

flops = result.get("depth_flops", 0.0)
params = result.get("depth_params", 0.0)
flops_g = flops / 1e9
params_m = params / 1e6

import torch
device_name = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"

st.markdown(f"""
<div class="metric-card" style="border-left: 3px solid var(--accent);">
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1.5rem; margin-top: 0.5rem;">
        <div>
            <span class="metric-label" style="color: var(--accent); font-weight: 600;">1. Chẩn đoán bệnh (CNN)</span>
            <div class="metric-value">{t_diag:.3f} s</div>
            <span class="metric-sub">Tải ảnh: {t_img_load:.3f}s | Load model: {t_cls_load:.3f}s | Dự đoán: {t_cls_infer:.3f}s</span>
        </div>
        <div>
            <span class="metric-label" style="color: var(--accent); font-weight: 600;">2. Xử lý Geo-GradCAM (3D)</span>
            <div class="metric-value">{t_gradcam:.3f} s</div>
            <span class="metric-sub">Depth Load: {t_depth_load:.3f}s | Depth Infer: {t_depth_infer:.3f}s<br>CAM Calc: {t_cam_calc:.3f}s | 3D Render: {t_render:.3f}s</span>
        </div>
        <div>
            <span class="metric-label" style="color: var(--accent); font-weight: 600;">3. Khuyến nghị Groq API</span>
            <div class="metric-value">{t_groq:.3f} s</div>
            <span class="metric-sub">Thời gian gửi prompt & nhận kết quả từ Groq LLM</span>
        </div>
        <div>
            <span class="metric-label" style="color: var(--accent); font-weight: 600;">4. Depth Anything V2</span>
            <div class="metric-value">{flops_g:.2f} GFLOPs</div>
            <span class="metric-sub">Tham số: {params_m:.2f}M Params (đo bằng thop)</span>
        </div>
    </div>
    <hr class="divider" style="margin: 1.2rem 0 0.6rem 0;">
    <div style="display: flex; justify-content: space-between; font-family: var(--mono); font-size: 0.8rem; color: var(--muted);">
        <span>Tổng thời gian pipeline: <b style="color: var(--accent); font-size: 0.9rem;">{t_total:.3f} s</b></span>
        <span>Thiết bị chạy mô hình: <b>{device_name}</b></span>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Weather info ──────────────────────────────────────────────────────────────
weather_data     = result.get("weather")
location         = result.get("location")
weather_scenario = result.get("weather_scenario", "WS_NORMAL")

if weather_data:
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("## 🌦️ Weather Information")

    loc_str = ""
    if location:
        city = location.get("city") or f"{location.get('lat', '')}, {location.get('lon', '')}"
        country = location.get("country", "")
        loc_str = f"{city}, {country}" if country else city

    daily = weather_data.get("daily_forecast", [])
    daily_rows = "".join(
        f"<tr>"
        f"<td style='padding:3px 8px;'>{d['date']}</td>"
        f"<td style='padding:3px 8px;'>{d['temp_min']}–{d['temp_max']}°C</td>"
        f"<td style='padding:3px 8px;'>{d['humidity_max']}%</td>"
        f"<td style='padding:3px 8px;'>{d['rain_mm']} mm</td>"
        f"</tr>"
        for d in daily
    )

    st.markdown(f"""
    <div class="weather-card">
        <span class="wlabel">{'📍 ' + loc_str if loc_str else ''} &nbsp;|&nbsp; Kịch bản: {weather_scenario}</span><br><br>
        <b>Hiện tại:</b>
        {weather_data['current_temp']}°C &nbsp;·&nbsp;
        Độ ẩm {weather_data['current_humidity']}% &nbsp;·&nbsp;
        Mưa {weather_data['current_rain']} mm<br><br>
        <b>Dự báo 3 ngày tới:</b><br>
        <table style="border-collapse:collapse;font-size:0.82rem;margin-top:4px;">
            <thead><tr style="color:var(--muted);">
                <th style="padding:3px 8px;text-align:left;">Ngày</th>
                <th style="padding:3px 8px;text-align:left;">Nhiệt độ</th>
                <th style="padding:3px 8px;text-align:left;">Độ ẩm max</th>
                <th style="padding:3px 8px;text-align:left;">Lượng mưa</th>
            </tr></thead>
            <tbody>{daily_rows}</tbody>
        </table>
        <br><span style="color:var(--muted);">Tổng mưa 3 ngày: {weather_data.get('rain_mm_3day_total', 0):.1f} mm</span>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.info("Không lấy được dữ liệu thời tiết (bỏ qua hoặc gặp lỗi kết nối).")


# ── Recommendation ────────────────────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("## Treatment Recommendation")

rec = result.get("recommendation", {})
if rec.get("error"):
    st.warning(f"Recommender error: {rec['error']}")
else:
    rec_text = rec.get("recommendation_text", "No recommendation returned.")
    st.markdown(f'<div class="rec-box">{rec_text}</div>', unsafe_allow_html=True)

    # Hiển thị nguồn tham khảo (nếu có)
    source_titles = rec.get("source_titles", [])
    if source_titles:
        sources_html = "".join(f"<div>• {s}</div>" for s in source_titles)
        with st.expander("📚 Nguồn tham khảo", expanded=False):
            st.markdown(f'<div class="source-list">{sources_html}</div>', unsafe_allow_html=True)


# ── GradCAM visualisations ────────────────────────────────────────────────────
task_data = result.get("tasks", {})

if task_data:
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("## GradCAM Visualisations")

    for task_name, data in task_data.items():
        with st.expander(f"📊 {task_name.capitalize()} — class index {data.get('gradcam_class_idx', '?')}", expanded=True):
            col_2d, col_3d = st.columns(2)

            with col_2d:
                st.markdown("#### 2D Overlay")
                overlay_path = data.get("overlay_2d_path")
                if overlay_path and os.path.exists(overlay_path):
                    st.image(overlay_path, use_container_width=True)
                else:
                    st.caption("2D overlay not saved (disabled or path missing).")

            with col_3d:
                st.markdown("#### Geo-GradCAM (3D)")
                html_path = data.get("geo_gradcam_html")
                if html_path and os.path.exists(html_path):
                    with open(html_path, "r", encoding="utf-8") as f:
                        html_content = f.read()
                    st.components.v1.html(html_content, height=520, scrolling=False)

                    with open(html_path, "rb") as f:
                        st.download_button(
                            label="⬇ Download interactive HTML",
                            data=f,
                            file_name=os.path.basename(html_path),
                            mime="text/html",
                            key=f"dl_{task_name}",
                        )
                else:
                    st.caption("3D visualisation not saved (disabled or path missing).")
else:
    # task_results hiện tại trong pipeline chưa được điền (for-loop pass)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.info(
        "ℹ️ GradCAM visualisation chưa được tích hợp vào pipeline hiện tại. "
        "Implement phần Grad-CAM trong vòng lặp `for task_for_cam in tasks` của `pipeline.py` "
        "để kích hoạt tính năng này."
    )


# ── Raw JSON (collapsible) ────────────────────────────────────────────────────
with st.expander("🔍 Raw result JSON", expanded=False):
    import json
    # Remove non-serialisable items before dumping
    safe = {k: v for k, v in result.items() if k not in ("tasks", "recommendation")}
    # Recommendation: bỏ "prompt" (dài) và giữ phần còn lại
    rec_safe = {
        k: v for k, v in rec.items()
        if k not in ("prompt",)
    } if rec else {}
    safe["recommendation"] = rec_safe
    safe["tasks"] = {
        t: {k: v for k, v in d.items() if k not in ("geo_gradcam_html",)}
        for t, d in task_data.items()
    }
    st.code(json.dumps(safe, indent=2, default=str), language="json")
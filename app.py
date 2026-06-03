"""
Demo Streamlit — Nhận diện mũ bảo hiểm (best.onnx)
Chạy: streamlit run app.py
"""

from __future__ import annotations

import io
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from inference import (
    ALERT_CLASSES,
    CLASS_NAMES,
    get_alerts,
    has_alert,
    load_model,
    predict_frame,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nhận diện mũ bảo hiểm",
    page_icon="🪖",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).parent
DEFAULT_MODEL = ROOT / "best.onnx"

# Định dạng ảnh hỗ trợ (gồm .webp)
IMAGE_EXTENSIONS = ["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff", "gif"]

# Session state keys — upload & kết quả video
_STATE_UPLOAD = ("img_upload", "vid_upload")
_STATE_VIDEO = ("vid_key", "vid_out_path", "vid_stats")


def clear_all_data() -> None:
    """Xóa toàn bộ ảnh, video và kết quả đã xử lý."""
    for key in (*_STATE_UPLOAD, *_STATE_VIDEO):
        st.session_state.pop(key, None)


def load_image_bgr(file_bytes: bytes) -> np.ndarray | None:
    """Đọc ảnh từ bytes → BGR (OpenCV). Fallback PIL nếu cần."""
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is not None:
        return img
    try:
        pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        return None

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Cài đặt")
model_path = st.sidebar.text_input("Đường dẫn model ONNX", value=str(DEFAULT_MODEL))
conf_thresh = st.sidebar.slider("Ngưỡng confidence", 0.1, 0.95, 0.5, 0.05)
st.sidebar.markdown("---")
st.sidebar.markdown("**Các lớp nhận diện:**")
for name in CLASS_NAMES:
    icon = "🔴" if name in ALERT_CLASSES else "🟢"
    st.sidebar.markdown(f"{icon} `{name}`")
st.sidebar.markdown("---")
st.sidebar.caption(
    "🔴 = cảnh báo khi phát hiện\n\n"
    "Export ONNX từ `best.pt`:\n"
    "```python\nfrom ultralytics import YOLO\n"
    'YOLO("best.pt").export(format="onnx", imgsz=640)\n```'
)
st.sidebar.markdown("---")
if st.sidebar.button(
    "🗑️ Xóa hết dữ liệu",
    type="primary",
    width="stretch",
    help="Xóa toàn bộ ảnh, video và kết quả nhận diện",
):
    clear_all_data()
    st.toast("Đã xóa toàn bộ dữ liệu.", icon="✅")
    st.rerun()


@st.cache_resource(show_spinner="Đang tải model ONNX…")
def get_model(path: str):
    return load_model(path)


# ── Header ───────────────────────────────────────────────────────────────────
st.title("🪖 Hệ thống nhận diện mũ bảo hiểm")
st.markdown(
    "Demo sử dụng **YOLO ONNX** (`best.onnx`) — nhận diện ảnh & video, "
    "cảnh báo khi phát hiện người **không đội mũ** (`nohelmet`)."
)

try:
    model = get_model(model_path)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()


def show_alerts(counts: Counter, container=None):
    """Display alert banners."""
    box = container or st
    alerts = get_alerts(counts)
    if alerts:
        for msg in alerts:
            box.error(f"🚨 **CẢNH BÁO:** {msg}")
    else:
        box.success("✅ Không phát hiện vi phạm (không đội mũ).")


def show_detection_table(counts: Counter):
    if not counts:
        st.info("Không phát hiện đối tượng nào.")
        return
    cols = st.columns(len(counts))
    for i, (label, n) in enumerate(counts.most_common()):
        color = "🔴" if label in ALERT_CLASSES else "🟢"
        cols[i].metric(f"{color} {label}", n)


# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_img, tab_vid, tab_about = st.tabs(["📷 Nhận diện ảnh", "🎬 Nhận diện video", "ℹ️ Thông tin"])

# ═══════════════════════════════════════════════════════════════════════════
# TAB ẢNH
# ═══════════════════════════════════════════════════════════════════════════
with tab_img:
    uploaded_list = st.file_uploader(
        "Tải một hoặc nhiều ảnh (JPG, PNG, WEBP, BMP, TIFF, GIF…)",
        type=IMAGE_EXTENSIONS,
        accept_multiple_files=True,
        key="img_upload",
    )

    if uploaded_list:
        total = len(uploaded_list)
        all_counts: Counter = Counter()
        failed: list[str] = []

        if total > 1:
            st.info(f"Đã chọn **{total}** ảnh — xử lý lần lượt từng ảnh.")

        for idx, uploaded in enumerate(uploaded_list, start=1):
            file_bytes = uploaded.read()
            nparr = load_image_bgr(file_bytes)

            if nparr is None:
                failed.append(uploaded.name)
                continue

            title = uploaded.name
            if total > 1:
                st.markdown(f"### Ảnh {idx}/{total}: `{title}`")
            else:
                st.subheader(title)

            col1, col2 = st.columns(2)

            with st.spinner(f"Đang nhận diện {title}…"):
                annotated, detections, counts = predict_frame(
                    model, nparr, conf=conf_thresh
                )

            all_counts.update(counts)

            with col1:
                st.caption("Ảnh gốc")
                st.image(
                    Image.open(io.BytesIO(file_bytes)),
                    width="stretch",
                )

            with col2:
                st.caption("Kết quả nhận diện")
                st.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    width="stretch",
                )

            show_alerts(counts)
            show_detection_table(counts)

            with st.expander(f"Chi tiết detection — {title}"):
                if detections:
                    for d in detections:
                        st.write(
                            f"- **{d['label']}** — conf: {d['conf']:.2%} — "
                            f"box: {d['box']}"
                        )
                else:
                    st.write("Không có đối tượng nào.")

            if idx < total:
                st.divider()

        if failed:
            st.warning(
                "Không đọc được: "
                + ", ".join(f"`{n}`" for n in failed)
            )

        if total > 1:
            st.markdown("---")
            st.subheader("📊 Tổng hợp tất cả ảnh")
            st.caption(
                f"Đã xử lý {total - len(failed)}/{total} ảnh thành công."
            )
            show_alerts(all_counts)
            show_detection_table(all_counts)

# ═══════════════════════════════════════════════════════════════════════════
# TAB VIDEO
# ═══════════════════════════════════════════════════════════════════════════
with tab_vid:
    video_file = st.file_uploader(
        "Tải video lên (MP4, AVI, MOV)",
        type=["mp4", "avi", "mov", "mkv"],
        key="vid_upload",
    )
    max_frames = st.number_input(
        "Giới hạn số frame xử lý (0 = toàn bộ video)",
        min_value=0,
        value=0,
        step=100,
        help="Đặt giới hạn để demo nhanh hơn với video dài.",
    )

    if video_file:
        suffix = Path(video_file.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
            tmp_in.write(video_file.read())
            input_path = tmp_in.name

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            st.error("Không mở được video.")
            st.stop()

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        limit = total_frames if max_frames == 0 else min(max_frames, total_frames)

        st.info(f"Video: {width}×{height} @ {fps:.1f} FPS — {total_frames} frames")

        progress = st.progress(0, text="Đang xử lý video…")
        alert_placeholder = st.empty()
        frame_preview = st.empty()

        global_counts: Counter = Counter()
        alert_frames = 0
        processed = 0

        out_path = tempfile.mktemp(suffix=".mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        while processed < limit:
            ret, frame = cap.read()
            if not ret:
                break

            annotated, _, counts = predict_frame(model, frame, conf=conf_thresh)
            global_counts.update(counts)
            writer.write(annotated)
            processed += 1

            if has_alert(counts):
                alert_frames += 1
                alert_placeholder.error(
                    f"🚨 Frame {processed}: phát hiện không đội mũ! "
                    f"(Tổng {alert_frames} frame cảnh báo)"
                )

            progress.progress(
                processed / limit,
                text=f"Đang xử lý frame {processed}/{limit}…",
            )

            if processed % max(1, limit // 10) == 0 or processed == limit:
                frame_preview.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    caption=f"Frame {processed}/{limit}",
                    width="stretch",
                )

        cap.release()
        writer.release()
        progress.progress(1.0, text="Hoàn tất!")

        st.markdown("---")
        st.subheader("📊 Tổng kết video")
        c1, c2, c3 = st.columns(3)
        c1.metric("Frames đã xử lý", processed)
        c2.metric("Frames cảnh báo", alert_frames)
        c3.metric(
            "Tỷ lệ vi phạm",
            f"{100 * alert_frames / max(processed, 1):.1f}%",
        )

        show_alerts(global_counts)
        show_detection_table(global_counts)

        with open(out_path, "rb") as f:
            st.download_button(
                "⬇️ Tải video kết quả",
                data=f,
                file_name="ket_qua_nhan_dien.mp4",
                mime="video/mp4",
            )

# ═══════════════════════════════════════════════════════════════════════════
# TAB THÔNG TIN
# ═══════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown(
        """
### Mô tả
Ứng dụng demo nhận diện **mũ bảo hiểm**, **biển số**, **người đi xe máy**
và cảnh báo khi phát hiện **không đội mũ** (`nohelmet`).

### Cách chạy
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Model
- File: `best.onnx` (YOLO, 4 classes)
- Export từ weights đã train:
```python
from ultralytics import YOLO
YOLO("best.pt").export(format="onnx", imgsz=640)
```

### Cảnh báo
| Lớp | Hành vi |
|-----|---------|
| `nohelmet` | 🔴 Cảnh báo đỏ — vi phạm không đội mũ |
| `helmet` | 🟢 Bình thường |
| `motorcyclist` | 🟢 Thông tin |
| `licenseplate` | 🟢 Thông tin |
        """
    )

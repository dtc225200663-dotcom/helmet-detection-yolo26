"""
Export YOLOv11 best.pt sang ONNX để chạy với Streamlit app
Chạy: python scripts/export_onnx.py
"""

from ultralytics import YOLO

WEIGHTS = "weights/best.pt"
model = YOLO(WEIGHTS)

print("Exporting to ONNX...")
model.export(format="onnx", imgsz=640, dynamic=True)
print("✅ Saved: weights/best.onnx")
print("Đặt file best.onnx vào thư mục gốc cùng app.py để chạy demo.")

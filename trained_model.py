import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os

# -------------------------------
# Load YOLOv8 model (once)
model_path = "/home/bit-user/Downloads/Pen_annotation/runs/detect/train/weights/best.pt"
yolo_model = YOLO(model_path)

st.title("YOLOv8 Pen Tracker with Persistent Counting")

uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()
    pen_count_placeholder = st.empty()

    tracked_pen_ids = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv8 tracker with persistence
        results = yolo_model.track(source=frame, persist=True, stream=True, imgsz=640)

        for r in results:
            annotated_frame = r.plot()
            boxes = r.boxes

            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0])
                    if yolo_model.names[cls] == "pen":
                        tracker_id = int(box.id[0]) if hasattr(box, "id") else None
                        if tracker_id is not None:
                            tracked_pen_ids.add(tracker_id)

        pen_count_placeholder.write(f"🖊 Unique pens detected so far: {len(tracked_pen_ids)}")
        stframe.image(annotated_frame, channels="BGR", use_container_width=True)

    cap.release()
    os.remove(video_path)

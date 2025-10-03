import streamlit as st
from ultralytics import YOLO
import os
from PIL import Image

# -------------------------------
# 1️⃣ Load your trained YOLOv8 model
# Replace this path with your actual trained weights
model_path = "/home/bit-user/Downloads/Pen_annotation/runs/detect/train/weights/best.pt"
model = YOLO(model_path)

st.title("🖼 YOLOv8 Pen Detection Frontend")
st.write("Upload an image and YOLOv8 will detect pens.")

st.write("Model classes:", model.names)  # confirm your model class, should be {0: 'pen'}

# -------------------------------
# 2️⃣ Upload image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Create folder to save uploaded images
    save_dir = "/home/bit-user/Downloads/Pen_annotation/valid"
    os.makedirs(save_dir, exist_ok=True)

    # Save uploaded image
    image_path = os.path.join(save_dir, uploaded_file.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display uploaded image
    st.image(Image.open(image_path), caption="Uploaded Image", use_column_width=True)

    st.write("### Predictions")
    results = model.predict(source=image_path, imgsz=640)

    # Annotated image with bounding boxes
    annotated_frame = results[0].plot()
    st.image(annotated_frame, caption="Predicted Image", use_column_width=True)

    # -------------------------------
    # 3️⃣ Count and display detected pens
    boxes = results[0].boxes  # list of detected boxes
    pen_boxes = [box for box in boxes if model.names[int(box.cls)] == "pen"]
    st.write(f"🖊 Detected pens: {len(pen_boxes)}")

    # Optional: show bounding box details
    if len(pen_boxes) > 0:
        st.write("### Bounding Box Details [class, confidence, x1, y1, x2, y2]")
        for i, box in enumerate(pen_boxes):
            cls = model.names[int(box.cls)]
            conf = float(box.conf)
            coords = box.xyxy[0].tolist()  # x1, y1, x2, y2
            st.write(f"{i+1}. Class: {cls}, Confidence: {conf:.2f}, Coordinates: {coords}")


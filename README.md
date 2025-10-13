🖊️ Pen Annotation - YOLOv8 Object Detection

This project is a YOLOv8-based object detection app for detecting pens (or other objects) using a dataset created with RoboFlow/LabelImg.
It includes a Streamlit frontend so you can easily upload images and view predictions with bounding boxes.


🚀 Features

Train YOLOv8 model on custom dataset (train/val/test split).

Save model weights (best.pt).

Streamlit frontend for image upload & prediction.

Outputs bounding boxes with confidence scores.


📂 Project Structure

Pen_annotation/
 ├── train/                # training dataset (images + labels)
 
 ├── val/                  # validation dataset (images + labels)
 
 ├── test/                 # test dataset (images + labels)
 
 ├── runs/                 # YOLOv8 training results (weights saved here)
 
 │    └── detect/
 
 │         └── train/
 
 │              └── weights/
 
 │                   ├── best.pt   # trained model (use this for prediction)
 
 │                   └── last.pt
 
 ├── dataset.yaml          # YOLOv8 dataset configuration
 
 ├── app.py                # Streamlit frontend
 
 ├── yolov8n.pt            # Pretrained YOLOv8 model (base model)
 
 └── README.md             # Project documentation



⚙️ Installation
1. Clone the repository
git clone https://github.com/your-username/Pen_annotation.git
cd Pen_annotation

2. Create virtual environment (optional but recommended)
   python3 -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

3. Install dependencies
   pip install -r requirements.txt
   
If you don’t have a requirements.txt yet, here are the needed libraries:
ultralytics
streamlit
opencv-python
pillow
numpy

📊 Training YOLOv8

Train the model using your dataset:

yolo train data=dataset.yaml model=yolov8n.pt epochs=50 imgsz=640


This will save results in:

runs/detect/train/weights/best.pt

🖼️ Running the Frontend

Run the Streamlit app to make predictions:

streamlit run app.py


Upload an image.

See detection results with bounding boxes and confidence scores.


🧪 Prediction with CLI (optional)

Run inference directly from command line:

yolo predict model=runs/detect/train/weights/best.pt source=test/images


Results will be saved in:

runs/detect/predict/


📌 Notes

Update the model_path in app.py with the correct path to your trained weights:

model_path = "runs/detect/train/weights/best.pt"


Update dataset.yaml with the correct number of classes (nc) and class names (names).


✨ Future Improvements

Add webcam real-time detection in Streamlit.

Use SAM/DINO for auto-labeling and backbone improvements.

Deploy the app to cloud (Streamlit Cloud / Hugging Face Spaces).




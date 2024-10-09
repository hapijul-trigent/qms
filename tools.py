import os
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from supervision import Detections, BoxAnnotator, LabelAnnotator, ColorPalette
from PIL import Image

@st.cache_resource(show_spinner=False)
def load_yolo_model(model_path):
    """Load and return the YOLOv8 model."""
    return YOLO(model_path)


def detect(image, model):
    """Perform object detection and return annotated image with highest confidence box only."""
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    result = model(image)[0]
    detections = Detections.from_ultralytics(result)
   
    if detections.xyxy.any():
        detections = detections[np.array([True if detections.confidence.max() == confidence else False for confidence in detections.confidence])]
        annotated_image = image.copy()
        
        # Annotate the highest confidence box
        annotator = BoxAnnotator(color=ColorPalette.ROBOFLOW, thickness=5)
        label_annotator = LabelAnnotator(text_scale=2, color=ColorPalette.ROBOFLOW, text_thickness=5)
        labels = [
            f"{class_name} {confidence*100:.2f}%"
            for class_name, confidence
            in zip(detections['class_name'], detections.confidence)
        ]
        annotated_image = label_annotator.annotate(annotated_image, detections=detections, labels=labels)
        annotated_image = annotator.annotate(annotated_image, detections=detections)
        return annotated_image
    else:
        return image




def detect_shoulder(image, model):
    """Perform object detection and return annotated image with highest confidence box only."""
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    result = model(image)[0]   

    return result.plot()

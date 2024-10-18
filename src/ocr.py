import os
import base64
import requests
import json
import pandas as pd
from PIL import Image, ExifTags
from io import BytesIO
from ultralytics import YOLO
from datetime import datetime


GPT4V_ENDPOINT = "https://genai-trigent-openai.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-02-15-preview"
MODEL_PATH = '/content/Model_Pill_Bottle_Nano-25.pt'

# Function to encode PIL images to base64
def encode_pil_images_to_base64(pil_images):
    encoded_images = {}
    for view, img in pil_images.items():
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('ascii')
        encoded_images[view] = img_str
    return encoded_images

# Function to send images and prompt to GPT-4V for text extraction
def extract_text_from_images(encoded_images):
    global GPT4V_KEY
    image_urls = [
        {
            "type": "image_url",
            "image_url": {
                "view": view,
                "url": f"data:image/jpeg;base64,{encoded_image}"
            }
        }
        for view, encoded_image in encoded_images.items()
    ]

    # GPT-4V prompt
    prompt = """Your extraction task details..."""

    payload = {
        "messages": [
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": image_urls
            }
        ],
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 2000
    }

    headers = {
        "Content-Type": "application/json",
        "api-key": GPT4V_KEY,
    }

    start = datetime.now()
    try:
        response = requests.post(GPT4V_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()
        end = datetime.now()
        print('Time: ', end - start)
        return response.json()['choices'][0]['message']['content']
    except requests.RequestException as e:
        raise SystemExit(f"Failed to make the request. Error: {e}")

# Function to convert JSON string to DataFrame
def json_to_dataframe(json_content):
    json_data = json.loads(json_content.replace("```json", '').replace("```", ''))
    df = pd.DataFrame(list(json_data.items()), columns=['Key', 'Value'])
    return df

# Function to process PIL images with YOLO model and return cropped images
def process_pil_images_with_yolo(model_path, pil_images):
    model = YOLO(model_path)
    cropped_images = {}
    
    for view, img in pil_images.items():
        # Save image to buffer to use with YOLO model
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        buffered.seek(0)

        # Perform prediction
        result = model(buffered)[0]
        img = correct_image_orientation(img)

        # Loop through each detected object in the image
        for i, (box, cls) in enumerate(zip(result.boxes.xyxy, result.boxes.cls)):
            class_name = model.names[int(cls)]
            if class_name == 'Cytomatrix--PillBottle-Label':  # Replace with your relevant class
                x1, y1, x2, y2 = map(int, box)
                cropped_img = img.crop((x1, y1, x2, y2))
                cropped_images[view] = cropped_img  # Save cropped images for further processing
    return cropped_images

# Main function to handle PIL images input, processing, and text extraction
def extract_text_from_pil_images_streamlit(pil_images, model_path=MODEL_PATH):
    # Process images with YOLO and get cropped images
    cropped_images = process_pil_images_with_yolo(model_path, pil_images)
    
    # Encode images to base64
    encoded_images = encode_pil_images_to_base64(cropped_images)
    
    # Extract text from images using GPT-4V
    json_content = extract_text_from_images(encoded_images)
    
    # Convert JSON content to a DataFrame
    df = json_to_dataframe(json_content)
    
    return df

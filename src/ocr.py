import requests
import json
import pandas as pd
from time import sleep
import streamlit as st


prompt = """Task: Extract **all text and details** from medical bottle images captured from various angles (90°, 180°, 270°, 360°). The goal is to accurately identify and extract **all printed information**, including product names, ingredients, usage instructions, manufacturing details, and any other relevant text. Ensure the extracted data is combined into **one structured output**.

Images:
The provided images feature the same medical bottle captured from multiple perspectives. Extract and consolidate all text across images into a **single, structured format**.

### Requirements:

**1. Text Extraction:**
- Capture every printed detail, including:
  - **Product name**
  - **Medicinal ingredients** (including herbs, chemicals, and dosage forms)
  - **Non-medicinal ingredients**
  - **Indications / Product description**
  - **Directions for use**
  - **Manufacturer details** (phone, address, website, etc.)
  - **LOT **
  - **Expiry date  / EXP**
  - **Warnings / Cautions / Additional instructions**
  - **Symbols or other markings** (e.g., regulatory labels, certifications, NPN numbers)
  - **Address** or **contact information** printed on the bottle or packaging.

- Ensure **case sensitivity** is respected (e.g., "C" vs. "c").
- Extract all **letters, numbers, symbols, and special characters** with **accuracy**.
- **Handle multi-line text** and **wrapped text** seamlessly.

**2. Handling Complex Text:**
- **Differentiate between sections** (e.g., "Ingredients" vs. "Directions").
- Handle **text wrapping around curved surfaces** effectively.
- **Preserve formatting** to distinguish between structured lists and continuous text.

**3. Post-Processing:**
- **Verify extracted text** for accuracy and completeness. Flag missing or partially illegible text.
- If any portion of text is unclear, provide a **best-guess approximation** and **flag it for review**.
- Use **"None"** for missing, unreadable, or absent information.

### Final Output:

- **Combine all information** into a **structured JSON format** with appropriate key-value pairs.
- For each extracted field, include **all possible details**.
- Only give JSON output not other unnecessary information
- Example of the required JSON structure:
```json
{
  "product name": value,
  "description": value,
  'quantity': value,
  "medicinal ingredients": {ingredient: Quantity},
  "nonmedicinal ingredients": value,
  "directions": value,
  "manufacturer name": value,
  "manufacturer address": value,
  "manufacturer phone": value,
  "manufacturer website": value,
  "LOT": value,
  "expiry date": value,
  "additional markings": value,
  "additional information": value,
  "warnings": value
}```
"""


def extract_text_from_base64_images(
        base64_images, 
        prompt,
        GPT4V_KEY,
        GPT4V_ENDPOINT="https://genai-trigent-openai.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-02-15-preview"):
    """
    Extracts text from base64-encoded images using GPT-4V and returns a DataFrame containing the extracted information.
    In case of a JSON decoding error, it retries up to 2 times.

    Parameters:
    - base64_images (dict): A dictionary with views (e.g., 'Front View', 'Back View') as keys and base64-encoded image data as values.

    Returns:
    - DataFrame: A Pandas DataFrame with extracted key-value pairs representing the text details from the images.
    """
    image_urls = [
        {
            "type": "image_url",
            "image_url": {
                "view": view,
                "url": f"data:image/jpeg;base64,{base64_img}"
            }
        }
        for view, base64_img in base64_images.items()
    ]
    

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
        "temperature": 0.3,
        "top_p": 0.95,
        "max_tokens": 3000
    }

    headers = {
        "Content-Type": "application/json",
        "api-key": GPT4V_KEY,
    }

    retries = 2
    for attempt in range(retries + 1):
        try:
            response = requests.post(GPT4V_ENDPOINT, headers=headers, json=payload)
            response.raise_for_status()
            json_content = response.json()['choices'][0]['message']['content']
            json_data = json.loads(json_content.replace("```json", '').replace("```", ''))
            # st.json(json_data)
            df = pd.DataFrame(list(json_data.items()), columns=['Key', 'Value'])
            return df
        except (json.JSONDecodeError, KeyError) as e:
            if attempt < retries:
                sleep(3)  # Delay between retries
                print(f"Retry {attempt + 1}/{retries} after JSONDecodeError: {e}")
            else:
                raise SystemExit(f"Failed to decode the JSON response after {retries} attempts. Error: {e}")
        except requests.RequestException as e:
            raise SystemExit(f"Failed to make the request. Error: {e}")

# QMS
**QMS** is a Quality Management System (QMS) tool developed for conducting visual inspections and quality assurance checks on products, specifically medical bottles. The application leverages image recognition, text extraction, and PDF report generation to automate and streamline the QA process. 

### Features

1. **Multi-view Image Analysis**:
   - Supports top, bottom, front, back, left, and right view image uploads.
   - Detects product details such as cap type, product type, and other visual features.

2. **Product Type Classification**:
   - Utilizes pre-trained YOLO models for classifying medical bottles into categories like dropper bottles, powder bottles, pill bottles, and liquid bottles.

3. **OCR (Optical Character Recognition)**:
   - Extracts text from images for detailed information retrieval, including product names, ingredients, and other important label information.

4. **PDF Report Generation**:
   - Generates structured reports based on the detected visual features and extracted text.
   - Reports are downloadable in PDF format and can be used for documentation and compliance purposes.

### Installation

1. Clone the repository:

   ```bash
   git clone https://bitbucket.org/hapijul-trigent/qms.git
   cd qms
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables by creating a `.env` file and adding necessary API keys like `GPT4V_KEY`:

   ```
   GPT4V_KEY=<your_api_key_here>
   ```

### Usage

To run the application, execute the following command:

```bash
streamlit run app.py
```

- Upload images of the product from different views (top, bottom, side views).
- The application will analyze the images and display detected features in real-time.
- You can download the final report in PDF format after the analysis is complete.

### Directory Structure

```
qms/
│
├── app.py                             # Main application file (Streamlit app)
├── export.csv                         # Exported data (sample)
├── output.pdf                         # Sample PDF output
├── requirements.txt                   # Project dependencies
├── .env                               # Environment variables
├── static/                            # Static images used in the app
│   ├── CanPrev_4D-logo.jpg
│   ├── Canprev-Logo.png
├── weights/                           # YOLO models used for image recognition
│   ├── Model_Dropper_Bottle_Nano-25.pt
│   ├── Model_Liquid_Bottle_Nano_25.pt
│   ├── Model_Pill_Bottle_Nano-25.pt
│   ├── Model_Powder_Bottle_Nano-25.pt
│   ├── Top-Bottom-Checks-v2-40.pt
│   ├── model_side_view_qa.pt
│   ├── model_unopened_botle_type_classification.pt
├── src/                               # Source code for different modules
│   ├── checklist.py                   # Updates and maintains the QA checklist
│   ├── image_processing.py            # Image processing utilities
│   ├── ocr.py                         # Optical Character Recognition (OCR) logic
│   ├── report_generation.py           # PDF report generation
│   ├── styles.py                      # Custom styles for report or app
│   ├── tools.py                       # Utility functions for model handling
│   ├── utils.py                       # Utility functions for post-processing
└── README.md                          # Project documentation (this file)
```

### Key Components

- **app.py**: The main entry point for running the application using Streamlit. It handles file uploads, model loading, and displaying analysis results.
- **src/checklist.py**: Logic for updating and maintaining the quality checklist during analysis.
- **src/image_processing.py**: Functions for processing images, including correcting orientations and converting cropped images to base64.
- **src/ocr.py**: Integrates GPT-4V to extract text from product labels and handles post-processing of the extracted data.
- **src/report_generation.py**: Contains functions for generating PDF reports from the analysis data.

### Models

The following YOLO models are used for object detection:

- `weights/model_side_view_qa.pt`: Detects side view features.
- `weights/Top-Bottom-Checks-v2-40.pt`: Analyzes top and bottom views of the bottles.
- Other models are also available for specific types of medical bottles, such as pill bottles, liquid bottles, etc.

### Dependencies

- Python 3.10+
- Streamlit
- OpenCV
- YOLO (Ultralytics)
- GPT-4V API for text extraction
- FPDF, ReportLab for report generation

Refer to `requirements.txt` for the full list of dependencies.

import streamlit as st
from src.tools import load_yolo_model
from src.image_processing import (
    correct_image_orientation,
    top_view_checks,
    bottom_view_checks,
    side_view_checks,
    merge_side_view_analysis
)
from src.checklist import CHECKLIST, update_CHECKLIST, TOP_CHECKLIST, BOTTOM_CHECKLIST, SIDE_CHECKLIST, SIDE_CHECKS_MAP
from src.report_generation import generate_pdf
from src.styles import apply_styles
from PIL import Image
import io
import pandas as pd



favicon = Image.open("static/Canprev-Logo.png")
st.set_page_config(
    page_title="Canprev AI",
    page_icon=favicon,
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_styles()
st.image("static/CanPrev_4D-logo.jpg", width=800)

# Load models
model_side_QA = load_yolo_model('weights/model_side_view_qa.pt')
model_top_base_qa = load_yolo_model('weights/TopBaseCheck-50.pt')


st.title("Unopened - Product Inspection Dashboard")
st.subheader("Upload Images")

col1, col2 = st.columns(2)
with col1:
    top_view_img = st.file_uploader("Top View", type=["jpg", "png", "jpeg"])
    if top_view_img is not None:
        top_view_img = Image.open(top_view_img)
        top_view_img = correct_image_orientation(top_view_img)

with col2:
    bottom_view_img = st.file_uploader("Bottom View", type=["jpg", "png", "jpeg"])
    if bottom_view_img is not None:
        bottom_view_img = Image.open(bottom_view_img)
        bottom_view_img = correct_image_orientation(bottom_view_img)

col3, col4 = st.columns(2)
with col3:
    left_view_img = st.file_uploader("Left View", type=["jpg", "png", "jpeg"])
    if left_view_img is not None:
        left_view_img = Image.open(left_view_img)
        left_view_img = correct_image_orientation(left_view_img)

with col4:
    right_view_img = st.file_uploader("Right View", type=["jpg", "png", "jpeg"])
    if right_view_img is not None:
        right_view_img = Image.open(right_view_img)
        right_view_img = correct_image_orientation(right_view_img)

col5, col6 = st.columns(2)
with col5:
    front_view_img = st.file_uploader("Front View", type=["jpg", "png", "jpeg"])
    if front_view_img is not None:
        front_view_img = Image.open(front_view_img)
        front_view_img = correct_image_orientation(front_view_img)

with col6:
    back_view_img = st.file_uploader("Back View", type=["jpg", "png", "jpeg"])
    if back_view_img is not None:
        back_view_img = Image.open(back_view_img)
        back_view_img = correct_image_orientation(back_view_img)

st.divider()


top, bottom, left, right, front, back = st.columns(6)
with top: top_view_panel = st.empty()
with bottom: bottom_view_panel = st.empty()
with left: left_view_panel = st.empty()
with right: right_view_panel = st.empty()
with front: front_view_panel = st.empty()
with back: back_view_panel = st.empty()

def simple_dict_to_streamlit_table(data_dict):
    """
    Converts a simple key-value dictionary into a stylish table using Streamlit.
    """
    
    df = pd.DataFrame(list(data_dict.items()), columns=['Checks', 'Status'])
    st.dataframe(df, hide_index=True, use_container_width=True)


def display_update_checklist(results, view_name):
    global CHECKLIST
    if view_name == 'Side':
        cols = st.columns(4)
        nocol = 4
    else:
        cols = st.columns(2)
        nocol = 2
    
    idx = 0
    for key, value in results.items():
        
        col = cols[idx % nocol]
        actual_class = value[0]
        if view_name == 'Side':
            value = all(value)
            key = SIDE_CHECKS_MAP[key]
            # if value:
            #     col.checkbox(f"{key[0]}: {key[1]}", value=True, key=f"{view_name}_{key}")
            # else:
            #     col.checkbox(f"{key[0]}: {key[1]}", value=False, key=f"{view_name}_{key}")
            CHECKLIST[key[0]] =  key[1]
        else:
            value = value[0]
            # if value:
            #     col.checkbox(f"{key}: {value}", value=True, key=f"{view_name}_{key}")
            # else:
            #     col.checkbox(f"{key}: {value}", value=False, key=f"{view_name}_{key}")
            CHECKLIST[key] =  value
        idx += 1

st.divider()
side_images = {
    "Left": left_view_img,
    "Right": right_view_img,
    "Front": front_view_img,
    "Back": back_view_img
}

annotation_view_panels = {
    'Top': top_view_panel,
    'Bottom': bottom_view_panel,
    "Left": left_view_panel,
    "Right": right_view_panel,
    "Front": front_view_panel,
    "Back": back_view_panel
}

if top_view_img and bottom_view_img and left_view_img and right_view_img and front_view_img and back_view_img:
    st.subheader("Unopened Bottle Checklist")
    top_check, bottom_check = st.columns(2)

    with top_check:
        top_annotated_view = top_view_checks(top_view_img, model=model_top_base_qa)
        annotation_view_panels['Top'].image(top_annotated_view, channels='bgr')
        display_update_checklist(TOP_CHECKLIST, "Top")

    with bottom_check:
        bottom_annotated_view = bottom_view_checks(bottom_view_img, model=model_top_base_qa)
        annotation_view_panels['Bottom'].image(bottom_annotated_view, channels='bgr')
        display_update_checklist(BOTTOM_CHECKLIST, "Bottom")

    side_images = {
        "Left": left_view_img,
        "Right": right_view_img,
        "Front": front_view_img,
        "Back": back_view_img
    }
    merge_side_view_analysis(side_images, annotation_view_panels)
    display_update_checklist(SIDE_CHECKLIST, "Side")

    simple_dict_to_streamlit_table(CHECKLIST)
    reset_enable = False
# if reset_enable:
#     if st.button('Clear'):
#         reset_enable = False
#         top_view_img, bottom_view_img, left_view_img, right_view_img, front_view_img, back_view_img = None, None, None, None, None, None


download_enabled = all([top_view_img, bottom_view_img, any(side_images.values())])
pdf_download_button, docs_download_button = st.columns([1, 1], vertical_alignment='bottom')

with pdf_download_button:
    if download_enabled:
        pdf = generate_pdf(CHECKLIST)
        pdf_output = io.BytesIO()
        pdf.output(pdf_output)
        pdf_output.seek(0)
        st.download_button(label="Download PDF", data=pdf_output, file_name="inspection_report.pdf", mime="application/pdf")

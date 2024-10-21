import streamlit as st
import io
from PIL import Image
import supervision as sv
import pandas as pd
from src.tools import load_yolo_model
from src.report_generation import generate_pdf
from src.image_processing import correct_image_orientation, convert_cropped_images_to_base64
from src.utils import post_process_checks, process_medicinal_ingredients
from dotenv import load_dotenv
from src.ocr import extract_text_from_base64_images, prompt
import os
from pprint import pprint

load_dotenv()
GPT4V_KEY = os.getenv("GPT4V_KEY")


favicon = Image.open("static/Canprev-Logo.png")
st.set_page_config(
    page_title="Canprev AI",
    page_icon=favicon,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.image("static/CanPrev_4D-logo.jpg", width=400)
st.markdown("""
<style>

	.stTabs [data-baseweb="tab-list"] {
		gap: 3px;
    }

	.stTabs [data-baseweb="tab"] {
		height: 40px;
        white-space: pre-wrap;
		background-color: #13276F;
		border-radius: 4px 4px 0px 0px;
		gap: 1px;
		padding: 10px 2px 10px 2px;
        color: white;
    }

	.stTabs [aria-selected="true"] {
  		background-color: #FFFFFF;
        color: #13276F;
        border: 2px solid #13276F;
        border-bottom: none;
	}

</style>""", unsafe_allow_html=True)

# Load Model
model_side_QA = load_yolo_model('weights/model_side_view_qa.pt')
model_top_base_qa = load_yolo_model('weights/Top-Bottom-Checks-v2-40.pt')
model_unopened_botle_type_classification = load_yolo_model('weights/model_unopened_botle_type_classification.pt')

model_side_view_QA = {
    'dropper_bottle' : load_yolo_model('weights/Model_Dropper_Bottle_Nano-25.pt'),
    'powder_botle' : load_yolo_model('weights/Model_Powder_Bottle_Nano25.pt'),
    'pill_botle': load_yolo_model('weights/Model_Pill_Bottle_Nano-25.pt'),
    'liquid_botle': load_yolo_model('weights/Model_Liquid_Bottle_Nano_25.pt')
}


from collections import defaultdict

CHECKLIST = dict()
DETECTIONS = defaultdict(tuple)






def identify_product_type(image, model):
    """Will check product type for Side view analysis"""
    try:
        pass
    except Exception as e:
        pass




def top_view_checks(image, model):
    """Performs Top View Analysis"""

    result = model(image)[0]
    detections = sv.Detections.from_ultralytics(result)
    detections = detections[detections.confidence > 0.8]

    DETECTIONS['Top'] = {class_:confidence for class_, confidence in zip(detections.data['class_name'], detections.confidence)}

    return result.plot()



def side_view_checks(image, view_name, model):
    """Performs Side View Analysis"""
    global CHECKLIST, DETECTIONS

    result = model(image)[0]
    detections = sv.Detections.from_ultralytics(result)
    detections = detections[detections.confidence > .7]
    
    DETECTIONS[view_name] = {class_:confidence for class_, confidence in zip(detections.data['class_name'], detections.confidence)}
    cropped_image = None
    for i, (box, cls) in enumerate(zip(result.boxes.xyxy, result.boxes.cls)):
            class_name = model.names[int(cls)]
            if 'Label' in class_name:
                print(class_name)
                x1, y1, x2, y2 = map(int, box)
                cropped_img = image.crop((x1, y1, x2, y2))
                # st.image(cropped_img)
    return result.plot(), cropped_img


def bottom_view_checks(image, model):
    """Performs Bottom View Checks"""

    result = model(image)[0]
    detections = sv.Detections.from_ultralytics(result)
    detections = detections[detections.confidence > .6]

    DETECTIONS['Bottom'] = {class_:confidence for class_, confidence in zip(detections.data['class_name'], detections.confidence)}

    return result.plot()



def merge_side_view_analysis(images, annotation_view_panels, model=None):
    """Analyze and Aggregate all side analysis"""
    
    cropped_label_images = {}
    model = model if model is not None else model_side_QA
    for view_name, image in images.items():
        if image:
            annotated_view, cropped_view_label = side_view_checks(image, view_name, model=model)
            annotation_view_panels[view_name].image(annotated_view, channels='bgr')
            cropped_label_images[view_name] = cropped_view_label
    return cropped_label_images





def clear_images():
    st.session_state['clear'] = True


st.title("QMS")
st.subheader("Upload Images")


if 'clear' not in st.session_state:
    st.session_state['clear'] = False

if 'report' not in st.session_state.keys():
    st.session_state['report'] = None

col1, col2 = st.columns(2)
if st.session_state['clear']:
    top_view_img, bottom_view_img, left_view_img, right_view_img, front_view_img, back_view_img = None, None, None, None, None, None

else:
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





if all([top_view_img, bottom_view_img, left_view_img, right_view_img, front_view_img, back_view_img]):
    st.divider()

    top, bottom, left, right, front, back = st.columns(6)
    with top: top_view_panel = st.empty()
    with bottom: bottom_view_panel = st.empty()
    with left: left_view_panel = st.empty()
    with right: right_view_panel = st.empty()
    with front: front_view_panel = st.empty()
    with back: back_view_panel = st.empty()

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
    st.subheader("QA Checks")
    top_check, bottom_check = st.columns(2)
    with st.spinner(text='Analyzing Cap'):
        with top_check:
            try:
                top_annotated_view = top_view_checks(top_view_img, model=model_top_base_qa)
                annotation_view_panels['Top'].image(top_annotated_view, channels='bgr')
                st.success('Analyzed Cap!')
            except Exception as e:
                st.error("Error Analyzing Cap : {e}")
        
    with st.spinner(text='Analyzing Base'):
        with bottom_check:
            try:
                bottom_annotated_view = bottom_view_checks(bottom_view_img, model=model_top_base_qa)
                annotation_view_panels['Bottom'].image(bottom_annotated_view, channels='bgr')
                st.success('Analyzed Base!')
            except Exception as e:
                   st.error("Error Analyzing Base : {e}")
    
    with st.spinner(text='Checking Product Type..'):
        product_type_results = model_unopened_botle_type_classification(front_view_img)[0]
        product_type = product_type_results.to_df().loc[0]['name']
        model = model_side_view_QA.get(product_type, None)
        CHECKLIST['Product Type'] = product_type.title().replace('_', ' ').replace('Botle', 'Bottle')
        
    with st.spinner(text='Analyzing Side Views'):
        cropped_label_images = merge_side_view_analysis(side_images, annotation_view_panels=annotation_view_panels, model=model)
        DETECTIONS, CHECKLIST, proces_checks_df = post_process_checks(DETECTIONS=DETECTIONS, CHECKLIST=CHECKLIST)
        st.success('Analyzed Side Views!')
    
    with st.spinner(text='Detecting Labels'):   
        bas64_label_images = convert_cropped_images_to_base64(cropped_images=cropped_label_images)
        st.session_state['bas64_label_images'] = bas64_label_images

        with st.container():
            checks_table, medicinal_ingredients = st.columns(2)
            with checks_table: st.dataframe(proces_checks_df, use_container_width=True, hide_index=True, height=400)
            with medicinal_ingredients: medicinal_ingredients_table = st.empty()

    with st.spinner("Analyzing Label Information"):

        df = extract_text_from_base64_images(base64_images=bas64_label_images, prompt=prompt, GPT4V_KEY=GPT4V_KEY)
        # path = '2024-10-18T10-27_export.csv'
        # df = pd.read_csv(path)
        df.columns = ['Label', 'Value']
        medicinal_df = process_medicinal_ingredients(df)
        if all(medicinal_df):
            df = df.loc[~(df.Label == 'medicinal ingredients')]
        else:
            pass
        st.dataframe(df, hide_index=True, use_container_width=True)
        medicinal_ingredients_table.dataframe(medicinal_df, hide_index=True, use_container_width=True, height=400)



    download_enabled = all([top_view_img, bottom_view_img, any(side_images.values())])
    pdf_download_button, docs_download_button = st.columns([1,1], vertical_alignment='bottom')
    with pdf_download_button:
        if download_enabled:
            pdf = generate_pdf(CHECKLIST)
            pdf_output = io.BytesIO()
            pdf.output(pdf_output)
            pdf_output.seek(0)
            st.download_button(label="Exprot Report", data=pdf_output, file_name="QA-Checklist.pdf", mime="application/pdf")
import streamlit as st
import io
from PIL import Image
import supervision as sv
import pandas as pd
from src.tools import load_yolo_model
from src.report_generation import generate_pdf
from src.checklist import update_CHECKLIST
from src.image_processing import correct_image_orientation
from dotenv import load_dotenv
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

TOP_CHECKLIST, SIDE_CHECKLIST, BOTTOM_CHECKLIST = defaultdict(list), defaultdict(list), defaultdict(list)
SIDE_CHECKS = {'label', 'neckband', 'shoulder', 'bottle'}
SIDE_CHECKS_MAP = {'label': ('Label', 'Present'), 'neckband': ('Neckband', 'Present'), 'shoulder': ('Shoulder', 'Curved'), 'bottle': ('Bottle', 'Good')}

SIDE_CHECKS_MAP_NEW = {
    'model_dropper_bottle_side_view_80': ['dropper_botle', 'dropper_botle_cap', 'dropper_botle_shoulder', 'label'],
    'model_powder_bottle_side_view': ['label', 'powder_botle', 'powder_botle_cap'],
    'model_liquid_bottle_side_view_80': ['Cytomatrix-Dermal-Liquid-Botle', 'Cytomatrix-Dermal-Liquid-Botle-Shoulder', 
            'Cytomatrix-Dermal-Liquid-Botle-With-Neckband', 'Magnesium_liquid_botle', 'Magnesium_liquid_botle_cap', 
            'Magnesium_liquid_botle_shoulder', 'label', 'Canprev-Gaba-Liquid-Botle', 
            'Canprev-Gaba-Liquid-Botle-Cap', 'Canprev-Omega-Liquid-Botle-Cap-With-Neckband', 
            'Canprev-Omega-Liquid-Botle', 'Canprev-Omega-Liquid-Botle-Shoulder', 'Canprev-Gaba-Liquid-Botle-shoulder'
        ],
}


TOP_CHECKS = {'Cap',}
TOP_CHECKS_MAP = {'Cap Type': 'Label Check', 'botle_with_neckband': 'Neckband Check', 'curved_shoulder': 'Shoulder Check'}
BOTTOM_CHECKS = {'Base',}
BOTTOM_CHECKS_MAP = {'Cap Type': 'Label Check', 'botle_with_neckband': 'Neckband Check', 'curved_shoulder': 'Shoulder Check'}





def identify_product_type(image, model):
    """Will check product type for Side view analysis"""
    try:
        pass
    except Exception as e:
        pass




def top_view_checks(image, model):
    
    result = model(image)[0]
    detections = sv.Detections.from_ultralytics(result)
    detections = detections[detections.confidence > 0.8]

    DETECTIONS['Top'] = {class_:confidence for class_, confidence in zip(detections.data['class_name'], detections.confidence)}

    if len(detections.xyxy) == 0:
        for thing in TOP_CHECKS:
            update_CHECKLIST(thing, False, TOP_CHECKLIST)
    else:
        for thing in TOP_CHECKS:
            update_CHECKLIST(thing, detections.data['class_name'][0], TOP_CHECKLIST)


    return result.plot()



def side_view_checks(image, view_name, model):
    """Performs Side View Checks"""
    global CHECKLIST, DETECTIONS

    result = model(image)[0]
    detections = sv.Detections.from_ultralytics(result)
    detections = detections[detections.confidence > .7]
    
    DETECTIONS[view_name] = {class_:confidence for class_, confidence in zip(detections.data['class_name'], detections.confidence)}

    
    if len(detections.xyxy) == 0:
        
        for thing in SIDE_CHECKS:
            update_CHECKLIST(thing, False, SIDE_CHECKLIST)
        print(SIDE_CHECKLIST, '\\\\')
    else:
        for thing in SIDE_CHECKS:

            if thing in detections.data['class_name']:
                update_CHECKLIST(thing, True, SIDE_CHECKLIST)
            else:
                update_CHECKLIST(thing, False, SIDE_CHECKLIST)
        
        if view_name == 'Front' and len(detections.data['class_name'][0].split('-')) == 4:
            try:
                
                for class_ in detections.data['class_name']:
                    
                    if ('sholder' in class_.lower() or 'shoulder' in class_.lower()):

                        print(class_.lower(), '----')
                        CHECKLIST['Shoulder'] = 'Good'
                        CHECKLIST['Shoulder Type'] = 'Curved' if 'curved' in class_.lower() else 'Flat'
                    
            except ValueError as e:
                
                CHECKLIST['Shoulder'] = None
                CHECKLIST['Shoulder Type'] = 'NA'
                


    return result.plot()


def bottom_view_checks(image, model):
    
    result = model(image)[0]
    detections = sv.Detections.from_ultralytics(result)
    detections = detections[detections.confidence > .6]

    DETECTIONS['Bottom'] = {class_:confidence for class_, confidence in zip(detections.data['class_name'], detections.confidence)}


    if len(detections.xyxy) == 0:
        for thing in BOTTOM_CHECKS:
            update_CHECKLIST(thing, False, BOTTOM_CHECKLIST)
    else:
        for thing in BOTTOM_CHECKS:
            update_CHECKLIST(thing, detections.data['class_name'][0], BOTTOM_CHECKLIST)


    return result.plot()




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
        actual_key = key

        if view_name == 'Side':
            print(value, '----')
            value = all(value)
            key = 'label' if 'label' in actual_key.lower() else (
                    'neckband' if 'neckband' in actual_key.lower() else (
                    'shoulder' if ('sholder' in actual_key.lower() or 'shoulder'in actual_key.lower()) else key
                )
            )
 
            key = SIDE_CHECKS_MAP[key]
            if value:
                if 'Curved'.casefold() in actual_key:
                    print(actual_key, '----------')
                CHECKLIST[key[0]] =  key[1]
        else:
            value = value[0]
            CHECKLIST[key] =  value
        idx += 1


def merge_side_view_analysis(images, annotation_view_panels, model=None):
    global CHECKLIST

    model = model if model is not None else model_side_QA
    for view_name, image in images.items():
        if image:
            annotated_view = side_view_checks(image, view_name, model=model)
            annotation_view_panels[view_name].image(annotated_view, channels='bgr')
            
    return True


# def simple_dict_to_streamlit_table(data_dict):
#     """
#     Converts a simple key-value dictionary into a stylish table using Streamlit.
#     """
#     global CHECKLIST
#     print(CHECKLIST)
    
#     if data_dict['Cap']:
#         brand, content_type, _, cap_type = data_dict['Cap'].split('-', 3)
#         data_dict.update({
#             'Product Brand': brand,
#             'Contains': content_type,
#             'Cap Type': cap_type.replace('-Cap', '').replace('Cap', 'Plastic')
#         })
#         data_dict.pop('Cap')
#     else:
#         data_dict.update({
#             'Product Brand': 'Unknown',
#             'Contains': 'Unknown',
#             'Cap Type': 'Unknown'
#         })
#         data_dict.pop('Cap')
#     # _, _, data_dict['Base'] = data_dict['Base'].split('-', 2) if 'Canprev-Type1-' not in data_dict['Base'] else (None, None, data_dict['Base'].replace('Canprev-Type1-', ''))
#     data_dict['Base'] = 'Good' if data_dict['Base'] else'Unknown'
#     data_dict['Cap'] = 'Present' if data_dict['Cap Type'] else 'Unknown'
#     if data_dict.get('Shoulder', None) or data_dict.get('CurvedShoulder', None):
#         data_dict['Shoulder Type'] = data_dict['Shoulder Type']
#         data_dict['Shoulder'] = 'Good' if data_dict['Shoulder'] else 'Unknown'
#     else:
#         data_dict['Shoulder Type'] = 'NA'
#         data_dict['Shoulder'] = 'NA'
    
#     if data_dict['Product Brand'].lower() == 'cytomatrix':
#         data_dict['Cap Pattern'] = 'Hexagon'
#     else:
#         data_dict['Cap Pattern'] = None

#     if data_dict['Product Type']== 'Dropper Bottle':
#         data_dict['Cap Type'] = 'Dropper'
#         data_dict['Contains'] =  'Dropper'


#     data_dict = {key: data_dict[key] for key in sorted(data_dict)}
#     CHECKLIST = data_dict
#     df = pd.DataFrame(list(data_dict.items()), columns=['Checks', 'Status'])
#     st.dataframe(df, hide_index=True, use_container_width=True)


def simple_dict_to_streamlit_table(data_dict):
    """
    Converts a simple key-value dictionary into a  table using Streamlit.
    """
    global DETECTIONS

    anomaly = False
    
    if DETECTIONS.get('Top', False):

        label = list(DETECTIONS.get('Top').keys())[0] 
        CHECKLIST['Cap'] = 'Good'
        CHECKLIST['Cap Pattern'] = 'Hexagon' if 'Cytomatrix' in label else 'Plain'
        CHECKLIST['Cap Type'] = 'Steel' if 'Steel' in label else 'Plastic' 
    
    else:

        CHECKLIST['Cap'] = 'Unknown'
        CHECKLIST['Cap Pattern'] = 'Unknown'
        CHECKLIST['Cap Pattern'] = 'Unknown'
    
    CHECKLIST['Base'] = 'Good' if DETECTIONS.get('Bottom', False) else 'Damaged'

    if (len(DETECTIONS.get('Left', [])) == 4) and (len(DETECTIONS.get('Right', [])) == 4) and (len(DETECTIONS.get('Front', [])) == 4) and (len(DETECTIONS.get('Back', [])) == 4):
        st.success('All Good')
        good_side_checks = set(DETECTIONS["Left"].keys()) & set(DETECTIONS["Right"].keys()) & set(DETECTIONS["Front"].keys()) & set(DETECTIONS["Back"].keys())
        for check in {'label', 'neckband', 'shoulder', 'bottle'}:
           
            if check in ' '.join(good_side_checks).lower():
                CHECKLIST[check.title()] = 'Good'
                if check == 'shoulder':
                    CHECKLIST['Shoulder Type'] = 'Curved' if 'curved' in ' '.join(good_side_checks).lower() else 'Flat'
            else:
                CHECKLIST[check.title()] = 'Unknown'
    else:
        st.error('Some Anomaly')
        anomaly = True

    
    if anomaly:
        pass
    else:
        df = pd.DataFrame(list(CHECKLIST.items()), columns=['Checks', 'Status'])
        st.dataframe(df, hide_index=True, use_container_width=True)




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
    st.subheader("Unopened Bottle Checklist")
    top_check, bottom_check = st.columns(2)

    with top_check:
        top_annotated_view = top_view_checks(top_view_img, model=model_top_base_qa)
        annotation_view_panels['Top'].image(top_annotated_view, channels='bgr')
        

    with bottom_check:
        bottom_annotated_view = bottom_view_checks(bottom_view_img, model=model_top_base_qa)
        annotation_view_panels['Bottom'].image(bottom_annotated_view, channels='bgr')
        

    product_type_results = model_unopened_botle_type_classification(front_view_img)[0]
    product_type = product_type_results.to_df().loc[0]['name']
    print(product_type)
    model = model_side_view_QA.get(product_type, None)
    CHECKLIST['Product Type'] = product_type.title().replace('_', ' ').replace('Botle', 'Bottle')

    merge_side_view_analysis(side_images, annotation_view_panels=annotation_view_panels, model=model)
    # display_update_checklist(SIDE_CHECKLIST, "Side")
    simple_dict_to_streamlit_table(CHECKLIST)
    # st.json(DETECTIONS)
    st.session_state['report'] = True



    download_enabled = all([top_view_img, bottom_view_img, any(side_images.values())])
    pdf_download_button, docs_download_button = st.columns([1,1], vertical_alignment='bottom')
    with pdf_download_button:
        if download_enabled:
            pdf = generate_pdf(CHECKLIST)
            pdf_output = io.BytesIO()
            pdf.output(pdf_output)
            pdf_output.seek(0)
            st.download_button(label="Download PDF", data=pdf_output, file_name="QA-Checklist.pdf", mime="application/pdf")

        # with docs_download_button:
        #     if download_enabled:
        #         doc = generate_docx(CHECKLIST)
        #         doc_output = io.BytesIO()
        #         doc.save(doc_output)
        #         doc_output.seek(0)

        #         st.download_button(label="Download DOCX", data=doc_output, file_name="inspection_report.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")


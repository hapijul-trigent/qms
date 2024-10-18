import streamlit as st
import pandas as pd

def post_process_checks(DETECTIONS, CHECKLIST):
    """
    Converts a simple key-value dictionary into a  table using Streamlit.
    """

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
        
        # st.checkbox('QA Checks', value=True)
        good_side_checks = set(DETECTIONS["Left"].keys()) & set(DETECTIONS["Right"].keys()) & set(DETECTIONS["Front"].keys()) & set(DETECTIONS["Back"].keys())
        for check in {'label', 'neckband', 'shoulder', 'bottle'}:
           
            if check in ' '.join(good_side_checks).lower():
                
                CHECKLIST[check.title()] = 'Good'
                if check == 'shoulder':
                    CHECKLIST['Shoulder Type'] = 'Curved' if 'curved' in ' '.join(good_side_checks).lower() else 'Flat'
            else:
                CHECKLIST[check.title()] = 'Unknown'
    else:
        # st.checkbox('QA Checks', value=False)
        st.error('Some Anomaly')
        anomaly = True

    
    if anomaly:
        pass
    else:
        df = pd.DataFrame(list(CHECKLIST.items()), columns=['Checks', 'Status'])
        st.dataframe(df, hide_index=True, use_container_width=True)
    
    return DETECTIONS, CHECKLIST
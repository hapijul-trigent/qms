import streamlit as st
import pandas as pd
import json

def post_process_checks(DETECTIONS, CHECKLIST):
    """
    Converts a simple key-value dictionary into a  table using Streamlit.
    """

    anomaly = False
    
    if DETECTIONS.get('Top', False):

        label = list(DETECTIONS.get('Top').keys())[0] 
        CHECKLIST['Cap'] = 'Present - Good'
        CHECKLIST['Cap Pattern'] = 'Hexagon' if 'Cytomatrix' in label else 'Plain'
        CHECKLIST['Cap Type'] = 'Steel' if 'Steel' in label else 'Plastic' 
    
    else:

        CHECKLIST['Cap'] = 'Unknown'
        CHECKLIST['Cap Pattern'] = 'Unknown'
        CHECKLIST['Cap Pattern'] = 'Unknown'
    
    CHECKLIST['Base'] = 'Good' if DETECTIONS.get('Bottom', False) else 'Damaged'

    if (len(DETECTIONS.get('Left', [])) == 4) and (len(DETECTIONS.get('Right', [])) == 4) and (len(DETECTIONS.get('Front', [])) == 4) and (len(DETECTIONS.get('Back', [])) == 4):
        
        
        good_side_checks = set(DETECTIONS["Left"].keys()) & set(DETECTIONS["Right"].keys()) & set(DETECTIONS["Front"].keys()) & set(DETECTIONS["Back"].keys())
        for check in {'label', 'neckband', 'shoulder', 'bottle'}:
           
            if check in ' '.join(good_side_checks).lower():
                
                CHECKLIST[check.title()] = 'Present - Good' if check in {'label', 'neckband'} else 'Good'
            
                if check == 'shoulder':
                    CHECKLIST['Shoulder Type'] = 'Curved' if 'curved' in ' '.join(good_side_checks).lower() else 'Flat'
            else:
                CHECKLIST[check.title()] = 'Unknown'
    
    elif ('Powder' in CHECKLIST['Product Type']):
        st.info('Powder Bottle')    
    else:
        # st.checkbox('QA Checks', value=False)
        st.error('Some Anomaly')
        anomaly = True

    
    if anomaly:
        df = pd.DataFrame(list(CHECKLIST.items()), columns=['Checks', 'Status'])
    else:
        df = pd.DataFrame(list(CHECKLIST.items()), columns=['Checks', 'Status'])
        # st.dataframe(df, hide_index=True, use_container_width=True)
    
    return DETECTIONS, CHECKLIST, df



def process_medicinal_ingredients(df):
    """
    This function reads a CSV file, extracts medicinal ingredients, and converts them into a DataFrame.
    
    Args:
        csv_file_path (str): The file path to the CSV file.
    
    Returns:
        pd.DataFrame: A DataFrame containing medicinal ingredients and their corresponding amounts.
    
    Raises:
        FileNotFoundError: If the CSV file is not found.
        KeyError: If the expected keys are not present in the DataFrame.
        ValueError: If the medicinal ingredients data cannot be parsed properly.
    """
    try:
        df = df.T
        df.columns = df.loc['Label'].to_list()
        df = df.drop('Label')
        
        medicinal_ingredients_data = df.loc['Value', 'medicinal ingredients']
        # st.dataframe(medicinal_ingredients_data)
        medicinal_ingredians_df = pd.DataFrame(
            {
                'Medicinal Ingredient': list(medicinal_ingredients_data.keys()),
                'Quantity': list(medicinal_ingredients_data.values())
            }
        )
        
        return medicinal_ingredians_df
    
    except FileNotFoundError as e:
        raise FileNotFoundError(f"The file at {df} was not found.") from e
    
    except KeyError as e:
        raise KeyError("The required 'Key' or 'Value' columns are missing in the CSV file.") from e
    
    except ValueError as e:
        # raise ValueError("Error parsing the 'medicinal ingredients' data. Check the format.") from e
        return None

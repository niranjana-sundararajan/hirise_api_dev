import pandas as pd
import requests
from pprint import pprint

def LBL_parser(label_url):

    # Get the data from the url
    req = requests.get(label_url, stream=True)

    lbl_file = req.text
    lbl_dict = {}
    lbl_dict['file_parameters'] = {}
    lbl_dict['image_map_parameters'] = {}
    lbl_dict['viewing_parameters'] = {}
    lbl_dict['timing_parameters'] = {}
    lbl_dict['other_parameters'] = {}
    
  

    # Select the parameters to save
    file_parameters = ["FILE_NAME", "REQUIRED_STORAGE_BYTES","MISSION_PHASE_NAME", "ORBIT_NUMBER"]
    image_map_parameters = ["CENTER_LATITUDE", "CENTER_LONGITUDE", "MAXIMUM_LATITUDE","MINIMUM_LATITUDE",
     "EASTERNMOST_LONGITUDE", "WESTERNMOST_LONGITUDE" ]

    viewing_parameters = ["INCIDENCE_ANGLE", "EMISSION_ANGLE","PHASE_ANGLE", "LOCAL_TIME",
    "SOLAR_LONGITUDE", "SUB_SOLAR_AZIMUTH","NORTH_AZIMUTH"]

    timing_parameters = ["MRO:OBSERVATION_START_TIME", "START_TIME", "SPACECRAFT_CLOCK_START_COUNT",
     "STOP_TIME", "SPACECRAFT_CLOCK_STOP_COUNT", "PRODUCT_CREATION_TIME"]
    
    other_parameters = ["SCALING_FACTOR", "OFFSET", "CENTER_FILTER_WAVELENGTH"]

    file_dict = dict.fromkeys(file_parameters, "")
    image_dict = dict.fromkeys(image_map_parameters, "")
    view_dict = dict.fromkeys(viewing_parameters,"")
    time_dict = dict.fromkeys(timing_parameters,"")
    other_dict = dict.fromkeys(other_parameters,"")

    for line in lbl_file.split('\n'):
        # Split the label data 
        if '=' in line:
            line_array = line.split('=')
            line_key = line_array[0].strip()
            line_value = line_array[1].strip()

            # Append required values in the respective df formats
            if line_key in file_parameters:
                file_dict[line_key]= line_value
            elif line_key in viewing_parameters:
                view_dict[line_key]= line_value
            elif line_key in timing_parameters:
                time_dict[line_key]= line_value
            elif line_key in other_parameters:
                other_dict[line_key]= line_value
            elif line_key in image_map_parameters:
                image_dict[line_key]= line_value 
                          
    lbl_dict["file_parameters"]= file_dict
    lbl_dict["image_map_parameters"]= image_dict
    lbl_dict["viewing_parameters"] = view_dict
    lbl_dict["timing_parameters"] = time_dict
    lbl_dict["other_parameters"] = other_dict   
     
    return lbl_dict
        

# pprint(LBL_parser("https://hirise-pds.lpl.arizona.edu/PDS/RDR/PSP/ORB_001300_001399/PSP_001336_1560/PSP_001336_1560_COLOR.LBL"))
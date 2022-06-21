from pyamg import test
import requests
from bs4 import BeautifulSoup
from pprint import pprint
import pandas as pd

from tqdm import tqdm

from datetime import datetime
import os, sys
from pathlib import Path
import pkg_resources
import shutil


dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))

if __package__ is None or __package__ == '':
    # uses current directory visibility
    import  hirise_imgs
    import utils
    CSV_FILE_PATH = './HIRISE_api/hirise/hirise_data.csv'
else:
    # uses current package visibility
    from . import  hirise_imgs
    from . import utils
    CSV_FILE_PATH = pkg_resources.resource_filename('hirise', 'hirise_data.csv')
class Image_Client:

    def download_database(self, folder_path):       
         shutil.copyfile(CSV_FILE_PATH, folder_path)

    def get_images(self,local_database_path = False, file_name = None, orbit_number = None, center_latitude= None, center_longitude= None, maximum_latitude= None, minimum_latitude = None, easternmost_longitude= None,westernmost_longitude= None ):
        """Function that querys the database and gets the image according to the given query"""
        # Read from the database
        if local_database_path:
            hirise_df = pd.read_csv(local_database_path)
        else:
            hirise_df = pd.read_csv(CSV_FILE_PATH)

        # List of all parameters values
        input_param_list = [file_name, orbit_number, center_latitude, center_longitude, maximum_latitude, minimum_latitude, easternmost_longitude,westernmost_longitude]
        
        # List of all available parameters values - entered by the user
        available_input_params = []
        [available_input_params.append(param) for param in input_param_list if param]

        # Dataframe of all queried values
        queried_rows_df = hirise_df[hirise_df.isin(available_input_params).any(axis = 1)]

        # Filenames extracted from the queried dataframe    
        f_names = [str(x) for x in queried_rows_df["FILE_NAME"]]

        # Create HIRISE img object outputs using the hirise_img class
        Hirise_img_objects =  [hirise_imgs.HIRISEimage(f_name) for f_name in f_names]

        return Hirise_img_objects

    def filter_center_latlon(self,local_database_path = False,center_latlon = None,center_lat = None,center_lon = None):
        
        if local_database_path:
            hirise_df = pd.read_csv(local_database_path)
        else:
            hirise_df = pd.read_csv(CSV_FILE_PATH)
        
        if center_latlon != None:
            lat, lon = center_latlon
            queried_rows_df = hirise_df[(hirise_df["CENTER_LATITUDE"] == lat) & (hirise_df["CENTER_LONGITUDE"] == lon)]
        else:
            queried_rows_df = hirise_df[(hirise_df["CENTER_LATITUDE"] == center_lat) | (hirise_df["CENTER_LONGITUDE"] == center_lon)]

        # Filenames extracted from the queried dataframe    
        f_names = [str(x) for x in queried_rows_df["FILE_NAME"]]

        # Create HIRISE img object outputs using the hirise_img class
        Hirise_img_objects =  [hirise_imgs.HIRISEimage(f_name) for f_name in f_names]

        return Hirise_img_objects       

    def filter_latlon(self,maxmin_lat, eastwest_lon,local_database_path = False,):
        
        if local_database_path:
            hirise_df = pd.read_csv(local_database_path)
        else:
            hirise_df = pd.read_csv(CSV_FILE_PATH)

        east_lon, west_lon = eastwest_lon
        max_lat, min_lat = maxmin_lat
        queried_rows_df = hirise_df[(hirise_df["MAXIMUM_LATITUDE"] <= max_lat) & (hirise_df["MINIMUM_LATITUDE"].abs() >= min_lat)
        & (hirise_df["EASTERNMOST_LONGITUDE"] >= east_lon)& (hirise_df["WESTERNMOST_LONGITUDE"] >= west_lon)]


        # Filenames extracted from the queried dataframe    
        f_names = [str(x) for x in queried_rows_df["FILE_NAME"]]

        # Create HIRISE img object outputs using the hirise_img class
        Hirise_img_objects =  [hirise_imgs.HIRISEimage(f_name) for f_name in f_names]

        return Hirise_img_objects 

    def filter_local_time(self,local_database_path = False,time = None,round_to=0, time_range = None):
        # Read from the database
        
        if local_database_path:
            hirise_df = pd.read_csv(local_database_path)
        else:
            hirise_df = pd.read_csv(CSV_FILE_PATH)
        
        if time_range:
            t1, t2 = time_range
            queried_rows_df = hirise_df[(hirise_df["LOCAL_TIME"] > t1) & (hirise_df["LOCAL_TIME"] < t2)]
        else:
            queried_rows_df = hirise_df[hirise_df["LOCAL_TIME"].round(round_to) == time]

        # Filenames extracted from the queried dataframe    
        f_names = [str(x) for x in queried_rows_df["FILE_NAME"]]

        # Create HIRISE img object outputs using the hirise_img class
        Hirise_img_objects =  [hirise_imgs.HIRISEimage(f_name) for f_name in f_names]

        return Hirise_img_objects
    
    def filter_mission_phase(self,mission_phase,local_database_path = False):
        # Read from the database
        
        if local_database_path:
            hirise_df = pd.read_csv(local_database_path)
        else:
            hirise_df = pd.read_csv(CSV_FILE_PATH)

        queried_rows_df = hirise_df[hirise_df["MISSION_PHASE_NAME"] == mission_phase ]

        # Filenames extracted from the queried dataframe    
        f_names = [str(x) for x in queried_rows_df["FILE_NAME"]]

        # Create HIRISE img object outputs using the hirise_img class
        Hirise_img_objects =  [hirise_imgs.HIRISEimage(f_name) for f_name in f_names]

        return Hirise_img_objects   

    def filter_orbital_range(self,orbital_range, local_database_path = False):
        # Read from the database
        
        if local_database_path:
            hirise_df = pd.read_csv(local_database_path)
        else:
            hirise_df = pd.read_csv(CSV_FILE_PATH)

        queried_rows_df = hirise_df[hirise_df["MISSION_PHASE_NAME"] == orbital_range]

        # Filenames extracted from the queried dataframe    
        f_names = [str(x) for x in queried_rows_df["FILE_NAME"]]

        # Create HIRISE img object outputs using the hirise_img class
        Hirise_img_objects =  [hirise_imgs.HIRISEimage(f_name) for f_name in f_names]

        return Hirise_img_objects
        
    def reload_database(self,mission_phases, folder_path = None, append = True):
        """ Function that webscrapes and downloads the meta data into csv for the config files"""
        
        hirise_dataframe = pd.DataFrame(columns=["FILE_NAME", "MISSION_PHASE_NAME", "ORBIT_NUMBER",
        "ORBITAL_RANGE","CENTER_LATITUDE", "CENTER_LONGITUDE", "MAXIMUM_LATITUDE","MINIMUM_LATITUDE",
            "EASTERNMOST_LONGITUDE", "WESTERNMOST_LONGITUDE", "INCIDENCE_ANGLE", "EMISSION_ANGLE",
            "PHASE_ANGLE", "LOCAL_TIME","SOLAR_LONGITUDE", "SUB_SOLAR_AZIMUTH","NORTH_AZIMUTH", "MRO:OBSERVATION_START_TIME",
            "START_TIME", "SPACECRAFT_CLOCK_START_COUNT", "STOP_TIME", "SPACECRAFT_CLOCK_STOP_COUNT", "PRODUCT_CREATION_TIME",
            "SCALING_FACTOR", "OFFSET", "CENTER_FILTER_WAVELENGTH", "IMG_URL", "LABEL_URL"])
        
        if isinstance(mission_phases, str):
            mission_phases = [mission_phases]
        elif all(isinstance(item, str) for item in mission_phases): # check iterable for stringness of all items. Will raise TypeError if some_object is not iterable
            pass
        else:
            raise TypeError
             
        # Define the urls needed for the downloads
        base_url = 'https://hirise-pds.lpl.arizona.edu/PDS/RDR/'

        image_url_list = []  
        label_url_list = []

        # File Parameters
        file_name_list = []
        req_storage_list = []
        mission_phase_list = []
        orbital_number_list = []
        orbital_range_list = []

        # Image Map Parameters
        center_latitude_list = []
        center_longitude_list = []
        maximum_latitude_list = []
        minimum_latitude_list = []
        easternmost_longitude_list = []
        westernmost_longitude_list = []

        # Viewing Parameters
        incidence_angle_list = []
        emission_angle_list = []
        phase_angle_list = []
        local_time_list = []
        solar_longitude_list = []
        sub_solar_azimuth_list = []

        # Timimg Parameters
        north_azimuth_list = []
        ob_start_time_list = []
        start_time_list = []
        spacecraft_start_list = []
        stop_time_list = []
        spacecraft_stop_list = []
        product_creation_list = []

        # Other Parameters
        scaling_factor_list = []
        offset_list = []
        wavelength_list = []

        for mission_phase in tqdm(mission_phases):
            mission_url = base_url + str(mission_phase)

            # Get the data from the url
            req = requests.get(mission_url)

            # Use soup to create an HTML parser
            soup = BeautifulSoup(req.text, 'html.parser')

            # Find all the labels with the correct tag - "a"
            labels = soup.find_all("a")
            # List to save orbital labels
            orbital_labels = []

                    
            # Loop through all the orbial ranges
            for i in range(1,len(labels)):
                # Save orbital labels into a list
                orbital_labels.append(labels[i]['href'])

            # count the number of orbitals in the mission phase
            orbital_range = ""

            

            for orbits in orbital_labels:

                # Build the orbital url
                orbital_url = base_url + mission_phase + '/' + orbits

                # Get the data from the url
                req = requests.get(orbital_url)

                # Use soup to create an HTML parser
                soup = BeautifulSoup(req.text, 'html.parser')

                # Find all the labels with the correct tag - "a"
                labels = soup.find_all("a")

                # List to save observation labels
                observation_labels = []
                # Loop through all the orbial ranges
                for i in range(1,len(labels)):
                    # Save orbital labels into a list - for each orbital range
                    observation_labels.append(labels[i]['href'])

                
                for observation in observation_labels:
                    # Build the image and label url
                    label_url = base_url + mission_phase + '/' + orbits + observation + observation.split('/')[0] + "_COLOR.LBL"
                    label_url_list.append(label_url)
                    image_url = base_url + mission_phase + '/' + orbits + observation + observation.split('/')[0] + "_COLOR.JP2"
                    image_url_list.append(image_url)
                    orbital_range = str(orbits.split("_")[1].lstrip('0')) + " - " + str(orbits.split("_")[2].lstrip('0')[:-1])
                    orbital_range_list.append(orbital_range)
                    
                
              
        for label_url in tqdm(label_url_list):      
            try:
                parsed_label = utils.LBL_parser(label_url)
                # File Parameters
                file_params = parsed_label["file_parameters"]
                file_name_list.append(file_params["FILE_NAME"].strip('\"'))

                if file_params["REQUIRED_STORAGE_BYTES"]:
                    req_storage_list.append(int(file_params["REQUIRED_STORAGE_BYTES"].split("<")[0]))
                else :
                    req_storage_list.append(None)

                mission_phase_list.append(file_params["MISSION_PHASE_NAME"].strip('\"'))
                orbital_number_list.append(file_params["ORBIT_NUMBER"])
                

                # Image Map Parameters
                image_map_params = parsed_label["image_map_parameters"]
                if image_map_params["CENTER_LATITUDE"]:
                    center_latitude_list.append(float(image_map_params["CENTER_LATITUDE"].split("<")[0]))
                else:
                    center_latitude_list.append(None)

                if image_map_params["CENTER_LONGITUDE"]:
                    center_longitude_list.append(float(image_map_params["CENTER_LONGITUDE"].split("<")[0]))
                else:
                    center_longitude_list.append(None)

                if image_map_params["MAXIMUM_LATITUDE"]:
                    maximum_latitude_list.append(float(image_map_params["MAXIMUM_LATITUDE"].split("<")[0]))
                else:
                    maximum_latitude_list.append(None)

                if image_map_params["MINIMUM_LATITUDE"]:
                    minimum_latitude_list.append(float(image_map_params["MINIMUM_LATITUDE"].split("<")[0]))
                else:
                    minimum_latitude_list.append(None)

                if image_map_params["EASTERNMOST_LONGITUDE"]:
                    easternmost_longitude_list.append(float(image_map_params["EASTERNMOST_LONGITUDE"].split("<")[0]))
                else:
                    easternmost_longitude_list.append(None)

                if image_map_params["WESTERNMOST_LONGITUDE"]:
                    westernmost_longitude_list.append(float(image_map_params["WESTERNMOST_LONGITUDE"].split("<")[0]))
                else:
                    westernmost_longitude_list.append(None)

                # Viewing Parameters
                view_params = parsed_label["viewing_parameters"]

                if view_params["INCIDENCE_ANGLE"]:
                    incidence_angle_list.append(float(view_params["INCIDENCE_ANGLE"].split("<")[0]))
                else:
                    incidence_angle_list.append(None)

                if view_params["EMISSION_ANGLE"]:
                    emission_angle_list.append(float(view_params["EMISSION_ANGLE"].split("<")[0]))
                else:
                    emission_angle_list.append(None)

                if view_params["PHASE_ANGLE"]:
                    phase_angle_list.append(float(view_params["PHASE_ANGLE"].split("<")[0]))
                else:
                    phase_angle_list.append(None)

                if view_params["LOCAL_TIME"]:
                    local_time_list.append(float(view_params["LOCAL_TIME"].split("<")[0]))
                else:
                    local_time_list.append(None)
                
                if view_params["SOLAR_LONGITUDE"]:
                    solar_longitude_list.append(float(view_params["SOLAR_LONGITUDE"].split("<")[0]))
                else:
                    solar_longitude_list.append(None)

                if view_params["SUB_SOLAR_AZIMUTH"]:
                    sub_solar_azimuth_list.append(float(view_params["SUB_SOLAR_AZIMUTH"].split("<")[0]))
                else:
                    sub_solar_azimuth_list.append(None)

                if view_params["NORTH_AZIMUTH"]:
                    north_azimuth_list.append(float(view_params["NORTH_AZIMUTH"].split("<")[0]))
                else:
                    north_azimuth_list.append(None)

                # Timimg Parameters
                timing_params = parsed_label["timing_parameters"]
                ob_start_time_list.append(timing_params["MRO:OBSERVATION_START_TIME"])
                start_time_list.append(timing_params["START_TIME"])
                spacecraft_start_list.append(timing_params["SPACECRAFT_CLOCK_START_COUNT"])
                stop_time_list.append(timing_params["STOP_TIME"])
                spacecraft_stop_list.append(timing_params["SPACECRAFT_CLOCK_STOP_COUNT"])
                product_creation_list.append(timing_params["PRODUCT_CREATION_TIME"])

                # Other Parameters
                other_params = parsed_label["other_parameters"]

                if view_params["NORTH_AZIMUTH"]:
                    scaling_factor_list.append(float(other_params["SCALING_FACTOR"]))
                else:
                    scaling_factor_list.append(None)

                if view_params["NORTH_AZIMUTH"]:
                    offset_list.append(float(other_params["OFFSET"]))
                else:
                    offset_list.append(None)

                wavelength_list.append(other_params["CENTER_FILTER_WAVELENGTH"])
            except:
                pass          
            # Add lists to dataframe
        hirise_dataframe["FILE_NAME"] = file_name_list
        hirise_dataframe["REQUIRED_STORAGE_BYTES"] = req_storage_list
        hirise_dataframe["MISSION_PHASE_NAME"] = mission_phase_list
        hirise_dataframe["ORBIT_NUMBER"] = orbital_number_list
        hirise_dataframe["ORBITAL_RANGE"] = orbital_range_list

        hirise_dataframe["CENTER_LATITUDE"] = center_latitude_list
        hirise_dataframe["CENTER_LONGITUDE"] = center_longitude_list
        hirise_dataframe["MAXIMUM_LATITUDE"] = maximum_latitude_list
        hirise_dataframe["MINIMUM_LATITUDE"] = minimum_latitude_list
        hirise_dataframe["EASTERNMOST_LONGITUDE"] = easternmost_longitude_list
        hirise_dataframe["WESTERNMOST_LONGITUDE"] = westernmost_longitude_list

        hirise_dataframe["INCIDENCE_ANGLE"] = incidence_angle_list
        hirise_dataframe["EMISSION_ANGLE"]= emission_angle_list
        hirise_dataframe["PHASE_ANGLE"] = phase_angle_list
        hirise_dataframe["LOCAL_TIME"] = local_time_list
        hirise_dataframe["SOLAR_LONGITUDE"] = solar_longitude_list
        hirise_dataframe["SUB_SOLAR_AZIMUTH"] = sub_solar_azimuth_list
        hirise_dataframe["NORTH_AZIMUTH"] = north_azimuth_list

        hirise_dataframe["MRO:OBSERVATION_START_TIME"] = ob_start_time_list
        hirise_dataframe["START_TIME"] = start_time_list
        hirise_dataframe["SPACECRAFT_CLOCK_START_COUNT"] = spacecraft_start_list
        hirise_dataframe["STOP_TIME"] = product_creation_list
        hirise_dataframe["SPACECRAFT_CLOCK_STOP_COUNT"] = spacecraft_stop_list
        hirise_dataframe["PRODUCT_CREATION_TIME"] = product_creation_list

        hirise_dataframe["SCALING_FACTOR"] = scaling_factor_list
        hirise_dataframe["OFFSET"] = offset_list
        hirise_dataframe["CENTER_FILTER_WAVELENGTH"] = wavelength_list
        hirise_dataframe["IMG_URL"] = image_url_list
        hirise_dataframe["LABEL_URL"] = label_url_list

        if __package__ is None or __package__ == '':
            hirise_dataframe.to_csv('hirise_data.csv',  encoding='utf-8')
        else:
            if append:
                 hirise_dataframe.to_csv(folder_path,mode='a',  encoding='utf-8')
            else:
                hirise_dataframe.to_csv(folder_path,  encoding='utf-8')
        
    def download(self,Hiriseimg_objs,folder_path, data_reload = True):
        if not folder_path:
            print(" Error! Please enter download path")
            return None
        if os.path.exists(folder_path) and data_reload:
            # Delete existing files in the directory
            [f.unlink() for f in Path(folder_path).glob("*") if f.is_file()]

            # Change into our current working directory - the newly made dir
            os.chdir(os.path.join(os.getcwd(), folder_path))

        else:
            # Make a directory inside the current working directory
            os.mkdir(os.path.join(os.getcwd(), folder_path)) 

            # Change into our current working directory - the newly made dir
            os.chdir(os.path.join(os.getcwd(), folder_path))
        if isinstance(Hiriseimg_objs, list):
            pass
        else:
            Hiriseimg_objs = [Hiriseimg_objs]

        for img in Hiriseimg_objs:
            image_url = img.get_img_url()

            with open(str(img.get_file_name()), 'wb' ) as file:
                # get the data
                image = requests.get(image_url)
                file.write(image.content)
                print('Saving file... ', img.get_file_name)

        # Return to parent directory
        sys.path.insert(0, parent_dir_path)



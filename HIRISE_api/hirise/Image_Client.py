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

# Define the current and parent directories and paths
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))

if __package__ is None or __package__ == "":
    # uses current directory visibility
    import Hirise_Image
    import utils

    CSV_FILE_PATH = "./HIRISE_api/hirise/hirise_data.csv"
else:
    # uses current package visibility
    from . import Hirise_Image
    from . import utils

    CSV_FILE_PATH = pkg_resources.resource_filename("hirise", "hirise_data.csv")


class Image_Client:
    def download_database(self, folder_path):
        shutil.copyfile(CSV_FILE_PATH, folder_path)

    def get_images(
        self,
        local_database_path=False,
        file_name=None,
        orbit_number=None,
        center_latitude=None,
        center_longitude=None,
        maximum_latitude=None,
        minimum_latitude=None,
        easternmost_longitude=None,
        westernmost_longitude=None,
    ):
        """Function that querys the database and gets the image according to the given query"""
        # Read from the database
        if local_database_path:
            hirise_df = pd.read_csv(local_database_path)
        else:
            hirise_df = pd.read_csv(CSV_FILE_PATH)

        # List of all parameters values
        input_param_list = [
            file_name,
            orbit_number,
            center_latitude,
            center_longitude,
            maximum_latitude,
            minimum_latitude,
            easternmost_longitude,
            westernmost_longitude,
        ]

        # List of all available parameters values - entered by the user
        available_input_params = []
        [available_input_params.append(param) for param in input_param_list if param]

        # Dataframe of all queried values
        queried_rows_df = hirise_df[hirise_df.isin(available_input_params).any(axis=1)]

        # Filenames extracted from the queried dataframe
        f_names = [str(x) for x in queried_rows_df["FILE_NAME"]]

        # Create HIRISE img object outputs using the hirise_img class
        Hirise_img_objects = [Hirise_Image.Hirise_Image(f_name) for f_name in f_names]

        return Hirise_img_objects

    def filter_center_latlon(
        self,
        local_database_path=False,
        center_latlon=None,
        center_lat=None,
        center_lon=None,
    ):

        if local_database_path:
            hirise_df = pd.read_csv(local_database_path)
        else:
            hirise_df = pd.read_csv(CSV_FILE_PATH)

        if center_latlon != None:
            lat, lon = center_latlon
            queried_rows_df = hirise_df[
                (hirise_df["CENTER_LATITUDE"] == lat)
                & (hirise_df["CENTER_LONGITUDE"] == lon)
            ]
        else:
            queried_rows_df = hirise_df[
                (hirise_df["CENTER_LATITUDE"] == center_lat)
                | (hirise_df["CENTER_LONGITUDE"] == center_lon)
            ]

        # Filenames extracted from the queried dataframe
        f_names = [str(x) for x in queried_rows_df["FILE_NAME"]]

        # Create HIRISE img object outputs using the hirise_img class
        Hirise_img_objects = [Hirise_Image.Hirise_Image(f_name) for f_name in f_names]

        return Hirise_img_objects

    def filter_latlon(
        self,
        maxmin_lat,
        eastwest_lon,
        local_database_path=False,
    ):

        if local_database_path:
            hirise_df = pd.read_csv(local_database_path)
        else:
            hirise_df = pd.read_csv(CSV_FILE_PATH)

        east_lon, west_lon = eastwest_lon
        max_lat, min_lat = maxmin_lat
        queried_rows_df = hirise_df[
            (hirise_df["MAXIMUM_LATITUDE"] <= max_lat)
            & (hirise_df["MINIMUM_LATITUDE"].abs() >= min_lat)
            & (hirise_df["EASTERNMOST_LONGITUDE"] >= east_lon)
            & (hirise_df["WESTERNMOST_LONGITUDE"] >= west_lon)
        ]

        # Filenames extracted from the queried dataframe
        f_names = [str(x) for x in queried_rows_df["FILE_NAME"]]

        # Create HIRISE img object outputs using the hirise_img class
        Hirise_img_objects = [Hirise_Image.Hirise_Image(f_name) for f_name in f_names]

        return Hirise_img_objects

    def filter_local_time(
        self, local_database_path=False, time=None, round_to=0, time_range=None
    ):
        # Read from the database

        if local_database_path:
            hirise_df = pd.read_csv(local_database_path)
        else:
            hirise_df = pd.read_csv(CSV_FILE_PATH)

        if time_range:
            t1, t2 = time_range
            queried_rows_df = hirise_df[
                (hirise_df["LOCAL_TIME"] > t1) & (hirise_df["LOCAL_TIME"] < t2)
            ]
        else:
            queried_rows_df = hirise_df[hirise_df["LOCAL_TIME"].round(round_to) == time]

        # Filenames extracted from the queried dataframe
        f_names = [str(x) for x in queried_rows_df["FILE_NAME"]]

        # Create HIRISE img object outputs using the hirise_img class
        Hirise_img_objects = [Hirise_Image.Hirise_Image(f_name) for f_name in f_names]

        return Hirise_img_objects

    def filter_mission_phase(self, mission_phase, local_database_path=False):
        # Read from the database

        if local_database_path:
            hirise_df = pd.read_csv(local_database_path)
        else:
            hirise_df = pd.read_csv(CSV_FILE_PATH)

        queried_rows_df = hirise_df[hirise_df["MISSION_PHASE_NAME"] == mission_phase]

        # Filenames extracted from the queried dataframe
        f_names = [str(x) for x in queried_rows_df["FILE_NAME"]]

        # Create HIRISE img object outputs using the hirise_img class
        Hirise_img_objects = [Hirise_Image.Hirise_Image(f_name) for f_name in f_names]

        return Hirise_img_objects

    def filter_orbital_range(self, orbital_range, local_database_path=False):
        # Read from the database

        if local_database_path:
            hirise_df = pd.read_csv(local_database_path)
        else:
            hirise_df = pd.read_csv(CSV_FILE_PATH)

        queried_rows_df = hirise_df[hirise_df["MISSION_PHASE_NAME"] == orbital_range]

        # Filenames extracted from the queried dataframe
        f_names = [str(x) for x in queried_rows_df["FILE_NAME"]]

        # Create HIRISE img object outputs using the hirise_img class
        Hirise_img_objects = [Hirise_Image.Hirise_Image(f_name) for f_name in f_names]

        return Hirise_img_objects

    def reload_database(
        self, mission_phases, folder_path=None, append=True, download_batch_size=None
    ):
        """Function that webscrapes and downloads the meta data into csv for the config files"""

        cols = [
            "FILE_NAME",
            "MISSION_PHASE_NAME",
            "ORBIT_NUMBER",
            "ORBITAL_RANGE",
            "CENTER_LATITUDE",
            "CENTER_LONGITUDE",
            "MAXIMUM_LATITUDE",
            "MINIMUM_LATITUDE",
            "EASTERNMOST_LONGITUDE",
            "WESTERNMOST_LONGITUDE",
            "INCIDENCE_ANGLE",
            "EMISSION_ANGLE",
            "PHASE_ANGLE",
            "LOCAL_TIME",
            "SOLAR_LONGITUDE",
            "SUB_SOLAR_AZIMUTH",
            "NORTH_AZIMUTH",
            "MRO:OBSERVATION_START_TIME",
            "START_TIME",
            "SPACECRAFT_CLOCK_START_COUNT",
            "STOP_TIME",
            "SPACECRAFT_CLOCK_STOP_COUNT",
            "PRODUCT_CREATION_TIME",
            "SCALING_FACTOR",
            "OFFSET",
            "CENTER_FILTER_WAVELENGTH",
            "IMG_URL",
            "LABEL_URL",
        ]

        hirise_dataframe = pd.DataFrame(columns=cols)
        if not append:
            if __package__ is None or __package__ == "":
                hirise_dataframe.to_csv(
                    "hirise_data_TRA.csv", mode="a", encoding="utf-8", index=False
                )
            else:
                hirise_dataframe.to_csv(
                    folder_path, mode="a", encoding="utf-8", index=False
                )
        if append:
            # Read last line of CSV and get mission phase orbital range
            pass

        if isinstance(mission_phases, str):
            mission_phases = [mission_phases]
        elif all(
            isinstance(item, str) for item in mission_phases
        ):  # check iterable for stringness of all items. Will raise TypeError if item is not iterable
            pass
        else:
            raise TypeError

        # Define the urls needed for the downloads
        base_url = "https://hirise-pds.lpl.arizona.edu/PDS/RDR/"

        image_url_list = []
        label_url_list = []

        orbital_range_list = []

        # other_parameters_list_of_lists = [scaling_factor_list, offset_list, wavelength_list]

        for mission_phase in tqdm(mission_phases):

            orbital_labels, len_orbitals = utils.get_webite_data(
                base_url, mission_phase
            )

            for orbits in orbital_labels:

                observation_labels, len_observation_labels = utils.get_webite_data(
                    base_url, mission_phase, orbits
                )
                for observation in observation_labels:
                    # Build the image and label url
                    label_url = (
                        base_url
                        + mission_phase
                        + "/"
                        + orbits
                        + observation
                        + observation.split("/")[0]
                        + "_COLOR.LBL"
                    )
                    label_url_list.append(label_url)
                    image_url = (
                        base_url
                        + mission_phase
                        + "/"
                        + orbits
                        + observation
                        + observation.split("/")[0]
                        + "_COLOR.JP2"
                    )
                    image_url_list.append(image_url)
                    orbital_range = (
                        str(orbits.split("_")[1].lstrip("0"))
                        + " - "
                        + str(orbits.split("_")[2].lstrip("0")[:-1])
                    )
                    try:
                        orbital_range_list.append(orbital_range)
                    except:
                        orbital_range_list.append(None)

        download_range_list = []

        for i in utils.downloadRange(0, len(label_url_list), download_batch_size):
            download_range_list.append(i)

        for i in range(len(download_range_list) - 1):
            # File Parameters
            file_name_list = []
            req_storage_list = []
            mission_phase_list = []
            orbital_number_list = []
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
            for label_url in label_url_list[
                download_range_list[i] : download_range_list[i + 1]
            ]:

                try:
                    parsed_label = utils.LBL_parser(label_url)
                    # File Parameters
                    file_params = parsed_label["file_parameters"]
                    file_name_list.append(file_params["FILE_NAME"].strip('"'))
                    utils.validate_append_float_data(
                        file_params["REQUIRED_STORAGE_BYTES"], req_storage_list
                    )
                    mission_phase_list.append(
                        file_params["MISSION_PHASE_NAME"].strip('"')
                    )
                    orbital_number_list.append(file_params["ORBIT_NUMBER"])

                    # Image Map Parameters
                    image_map_params = parsed_label["image_map_parameters"]
                    utils.validate_append_float_data(
                        image_map_params["CENTER_LATITUDE"], center_latitude_list
                    )
                    utils.validate_append_float_data(
                        image_map_params["CENTER_LONGITUDE"], center_longitude_list
                    )
                    utils.validate_append_float_data(
                        image_map_params["MAXIMUM_LATITUDE"], maximum_latitude_list
                    )
                    utils.validate_append_float_data(
                        image_map_params["MINIMUM_LATITUDE"], minimum_latitude_list
                    )
                    utils.validate_append_float_data(
                        image_map_params["EASTERNMOST_LONGITUDE"],
                        easternmost_longitude_list,
                    )
                    utils.validate_append_float_data(
                        image_map_params["WESTERNMOST_LONGITUDE"],
                        westernmost_longitude_list,
                    )

                    # Viewing Parameters
                    view_params = parsed_label["viewing_parameters"]
                    utils.validate_append_float_data(
                        view_params["INCIDENCE_ANGLE"], incidence_angle_list
                    )
                    utils.validate_append_float_data(
                        view_params["EMISSION_ANGLE"], emission_angle_list
                    )
                    utils.validate_append_float_data(
                        view_params["PHASE_ANGLE"], phase_angle_list
                    )
                    utils.validate_append_float_data(
                        view_params["LOCAL_TIME"], local_time_list
                    )
                    utils.validate_append_float_data(
                        view_params["SOLAR_LONGITUDE"], solar_longitude_list
                    )
                    utils.validate_append_float_data(
                        view_params["SUB_SOLAR_AZIMUTH"], sub_solar_azimuth_list
                    )
                    utils.validate_append_float_data(
                        view_params["NORTH_AZIMUTH"], north_azimuth_list
                    )

                    # Timimg Parameters
                    timing_params = parsed_label["timing_parameters"]
                    ob_start_time_list.append(
                        timing_params["MRO:OBSERVATION_START_TIME"]
                    )
                    start_time_list.append(timing_params["START_TIME"])
                    spacecraft_start_list.append(
                        timing_params["SPACECRAFT_CLOCK_START_COUNT"]
                    )
                    stop_time_list.append(timing_params["STOP_TIME"])
                    spacecraft_stop_list.append(
                        timing_params["SPACECRAFT_CLOCK_STOP_COUNT"]
                    )
                    product_creation_list.append(timing_params["PRODUCT_CREATION_TIME"])

                    # Other Parameters
                    other_params = parsed_label["other_parameters"]
                    utils.append_float_data_without_strip(
                        other_params["SCALING_FACTOR"], scaling_factor_list
                    )
                    utils.append_float_data_without_strip(
                        other_params["OFFSET"], offset_list
                    )
                    wavelength_list.append(other_params["CENTER_FILTER_WAVELENGTH"])

                except:
                    print("Error appending to list")
            # Add lists to dataframe
            hirise_dataframe["FILE_NAME"] = file_name_list
            hirise_dataframe["REQUIRED_STORAGE_BYTES"] = req_storage_list
            hirise_dataframe["MISSION_PHASE_NAME"] = mission_phase_list
            hirise_dataframe["ORBIT_NUMBER"] = orbital_number_list
            hirise_dataframe["ORBITAL_RANGE"] = orbital_range_list[
                download_range_list[i] : download_range_list[i + 1]
            ]

            hirise_dataframe["CENTER_LATITUDE"] = center_latitude_list
            hirise_dataframe["CENTER_LONGITUDE"] = center_longitude_list
            hirise_dataframe["MAXIMUM_LATITUDE"] = maximum_latitude_list
            hirise_dataframe["MINIMUM_LATITUDE"] = minimum_latitude_list
            hirise_dataframe["EASTERNMOST_LONGITUDE"] = easternmost_longitude_list
            hirise_dataframe["WESTERNMOST_LONGITUDE"] = westernmost_longitude_list

            hirise_dataframe["INCIDENCE_ANGLE"] = incidence_angle_list
            hirise_dataframe["EMISSION_ANGLE"] = emission_angle_list
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
            hirise_dataframe["IMG_URL"] = image_url_list[
                download_range_list[i] : download_range_list[i + 1]
            ]
            hirise_dataframe["LABEL_URL"] = label_url_list[
                download_range_list[i] : download_range_list[i + 1]
            ]

            if __package__ is None or __package__ == "":
                hirise_dataframe.to_csv(
                    "hirise_data_TRA.csv",
                    mode="a",
                    header=False,
                    encoding="utf-8",
                    index=False,
                )
            else:
                hirise_dataframe.to_csv(
                    folder_path, mode="a", encoding="utf-8", header=False, index=False
                )

            hirise_dataframe = pd.DataFrame()

    def download(self, Hiriseimg_objs, folder_path, data_reload=True):
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

            with open(str(img.get_file_name()), "wb") as file:
                # get the data
                image = requests.get(image_url)
                file.write(image.content)
                print("Saving file... ", img.get_file_name)

        # Return to parent directory
        sys.path.insert(0, parent_dir_path)

    def download_random_images(
        self, fol_path, image_count=1, local_database_path=False
    ):
        if local_database_path:
            hirise_df = pd.read_csv(local_database_path)
        else:
            hirise_df = pd.read_csv(CSV_FILE_PATH)

        random_samples = hirise_df.sample(n=image_count)

        # Filenames extracted from the queried dataframe
        f_names = [str(x) for x in random_samples["FILE_NAME"]]

        # Create HIRISE img object outputs using the hirise_img class
        Hirise_img_objects = [Hirise_Image.Hirise_Image(f_name) for f_name in f_names]

        Image_Client.download(
            self, Hiriseimg_objs=Hirise_img_objects, folder_path=fol_path
        )


# imc = Image_Client()
# imc.download_random_images(fol_path="./download-data", image_count=10)

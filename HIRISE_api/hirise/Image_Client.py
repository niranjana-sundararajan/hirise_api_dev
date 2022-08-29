from bs4 import BeautifulSoup
from tqdm import tqdm
from pathlib import Path

import requests
import pandas as pd
import os
import sys
import pkg_resources
import shutil
import random

# --------------------------------------------------------------------------
# Define the current and parent directories and paths
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))

# --------------------------------------------------------------------------
if __package__ is None or __package__ == "":
    # uses current directory visibility
    import Hirise_Image
    import utils

    # Load CSV files
    CSV_FILE_PATH = "./HIRISE_api/hirise/hirise_data.csv"
    THEME_FILE_PATH = "./HIRISE_api/hirise/theme_data.csv"
else:
    # uses current package visibility
    from . import Hirise_Image
    from . import utils

    # Load CSV Files
    CSV_FILE_PATH = pkg_resources.resource_filename(
        "hirise", "hirise_data.csv"
    )
    THEME_FILE_PATH = pkg_resources.resource_filename(
        "hirise", "theme_data.csv"
    )

# --------------------------------------------------------------------------


class ImageClient:
    def download_database(self, folder_path):
        """Function that downloads the current database available
         in the HIRISE Package"""
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
        """Function that queries the database and gets the
        image according to the given query"""
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
        [
            available_input_params.append(param)
            for param in input_param_list
            if param
        ]

        # Dataframe of all queried values
        queried_rows_df = hirise_df[
            hirise_df.isin(available_input_params).any(axis=1)
        ]

        # Filenames extracted from the queried dataframe
        f_names = [str(x) for x in queried_rows_df["FILE_NAME"]]

        # Create HIRISE img object outputs using the hirise_img class
        Hirise_img_objects = [
            Hirise_Image.HiriseImage(f_name) for f_name in f_names
        ]

        return Hirise_img_objects

    # ------------------------------------------------------------------------
    # Filter Images Based on Image Parameters
    # ------------------------------------------------------------------------
    def filter_center_latlon(
        self,
        local_database_path=False,
        center_latlon=None,
        center_lat=None,
        center_lon=None,
    ):
        """Function that ret the current database available in
         the HIRISE Package"""
        if local_database_path:
            hirise_df = pd.read_csv(local_database_path)
        else:
            hirise_df = pd.read_csv(CSV_FILE_PATH)

        if center_latlon is not None:
            lat, lon = center_latlon
            queried_rows_df = hirise_df[
                (hirise_df["CENTER_LATITUDE"] == lat) & 
                (hirise_df["CENTER_LONGITUDE"] == lon)
            ]
        else:
            queried_rows_df = hirise_df[
                (hirise_df["CENTER_LATITUDE"] == center_lat)
                | (hirise_df["CENTER_LONGITUDE"] == center_lon)
            ]

        # Filenames extracted from the queried dataframe
        f_names = [str(x) for x in queried_rows_df["FILE_NAME"]]

        # Create HIRISE img object outputs using the hirise_img class
        Hirise_img_objects = [
            Hirise_Image.HiriseImage(f_name) for f_name in f_names
        ]

        return Hirise_img_objects

    def filter_latlon(
        self,
        maxmin_lat,
        eastwest_lon,
        local_database_path=False,
    ):
        """Function that returns HIRISE image object filtered
        based on maximum ranges of latitudes and/or east-west
        longitudes given by user"""
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
        Hirise_img_objects = [
            Hirise_Image.HiriseImage(f_name) for f_name in f_names
        ]

        return Hirise_img_objects

    def filter_local_time(
        self, local_database_path=False, time=None, round_to=0,
            time_range=None
    ):
        """Function that returns HIRISE image object filtered
        based on local time entered by the user"""
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
            queried_rows_df = hirise_df[
                hirise_df["LOCAL_TIME"].round(round_to) == time
            ]

        # Filenames extracted from the queried dataframe
        f_names = [str(x) for x in queried_rows_df["FILE_NAME"]]

        # Create HIRISE img object outputs using the hirise_img class
        Hirise_img_objects = [
            Hirise_Image.HiriseImage(f_name) for f_name in f_names
        ]

        return Hirise_img_objects

    def filter_mission_phase(self, mission_phase, local_database_path=False):
        """Function that returns HIRISE image object filtered
        based on mission phase(s) entered by the user"""

        # Read from the database

        if local_database_path:
            hirise_df = pd.read_csv(local_database_path)
        else:
            hirise_df = pd.read_csv(CSV_FILE_PATH)

        queried_rows_df = hirise_df[
            hirise_df["MISSION_PHASE_NAME"] == mission_phase
        ]

        # Filenames extracted from the queried dataframe
        f_names = [str(x) for x in queried_rows_df["FILE_NAME"]]

        # Create HIRISE img object outputs using the hirise_img class
        Hirise_img_objects = [
            Hirise_Image.HiriseImage(f_name) for f_name in f_names
        ]

        return Hirise_img_objects

    def filter_orbital_range(self, orbital_range, local_database_path=False):
        """Function that returns HIRISE image object filtered
        based on orbital range entered by the user"""
        # Read from the database

        if local_database_path:
            hirise_df = pd.read_csv(local_database_path)
        else:
            hirise_df = pd.read_csv(CSV_FILE_PATH)

        queried_rows_df = hirise_df[
            hirise_df["MISSION_PHASE_NAME"] == orbital_range
        ]

        # Filenames extracted from the queried dataframe
        f_names = [str(x) for x in queried_rows_df["FILE_NAME"]]

        # Create HIRISE img object outputs using the hirise_img class
        Hirise_img_objects = [
            Hirise_Image.HiriseImage(f_name) for f_name in f_names
        ]

        return Hirise_img_objects

    # ------------------------------------------------------------------------
    # Download Images and Reloads Local Database
    # ------------------------------------------------------------------------
    def reload_database(
        self,
        mission_phases,
        folder_path=None,
        append=True,
        download_batch_size=None,
    ):
        """Function that webscrapes and downloads the metadata
        into csv for the config files"""

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
                    "hirise_data_TRA.csv",
                    mode="a",
                    encoding="utf-8",
                    index=False,
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
        ):  # check iterable for stringness of all items. Will raise TypeError
            # if item is not iterable
            pass
        else:
            raise TypeError

        # Define the urls needed for the downloads
        base_url = "https://hirise-pds.lpl.arizona.edu/PDS/RDR/"

        image_url_list = []
        label_url_list = []

        orbital_range_list = []

        # other_parameters_list_of_lists = [scaling_factor_list, offset_list,
        # wavelength_list]

        for mission_phase in tqdm(mission_phases):

            orbital_labels, len_orbitals = utils.get_website_data(
                base_url, mission_phase
            )

            for orbits in orbital_labels:

                (
                    observation_labels,
                    len_observation_labels,
                ) = utils.get_website_data(base_url, mission_phase, orbits)
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
                    except BaseException:
                        orbital_range_list.append(None)

        download_range_list = []

        for i in utils.downloadRange(
            0, len(label_url_list), download_batch_size
        ):
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
                download_range_list[i]: download_range_list[i + 1]
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
                        image_map_params["CENTER_LATITUDE"],
                        center_latitude_list,
                    )
                    utils.validate_append_float_data(
                        image_map_params["CENTER_LONGITUDE"],
                        center_longitude_list,
                    )
                    utils.validate_append_float_data(
                        image_map_params["MAXIMUM_LATITUDE"],
                        maximum_latitude_list,
                    )
                    utils.validate_append_float_data(
                        image_map_params["MINIMUM_LATITUDE"],
                        minimum_latitude_list,
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
                        view_params["SUB_SOLAR_AZIMUTH"],
                        sub_solar_azimuth_list,
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
                    product_creation_list.append(
                        timing_params["PRODUCT_CREATION_TIME"]
                    )

                    # Other Parameters
                    other_params = parsed_label["other_parameters"]
                    utils.append_float_data_without_strip(
                        other_params["SCALING_FACTOR"], scaling_factor_list
                    )
                    utils.append_float_data_without_strip(
                        other_params["OFFSET"], offset_list
                    )
                    wavelength_list.append(
                        other_params["CENTER_FILTER_WAVELENGTH"]
                    )

                except BaseException:
                    print("Error appending to list")
            # Add lists to dataframe
            hirise_dataframe["FILE_NAME"] = file_name_list
            hirise_dataframe["REQUIRED_STORAGE_BYTES"] = req_storage_list
            hirise_dataframe["MISSION_PHASE_NAME"] = mission_phase_list
            hirise_dataframe["ORBIT_NUMBER"] = orbital_number_list
            hirise_dataframe["ORBITAL_RANGE"] = orbital_range_list[
                download_range_list[i]: download_range_list[i + 1]
            ]

            hirise_dataframe["CENTER_LATITUDE"] = center_latitude_list
            hirise_dataframe["CENTER_LONGITUDE"] = center_longitude_list
            hirise_dataframe["MAXIMUM_LATITUDE"] = maximum_latitude_list
            hirise_dataframe["MINIMUM_LATITUDE"] = minimum_latitude_list
            hirise_dataframe[
                "EASTERNMOST_LONGITUDE"
            ] = easternmost_longitude_list
            hirise_dataframe[
                "WESTERNMOST_LONGITUDE"
            ] = westernmost_longitude_list

            hirise_dataframe["INCIDENCE_ANGLE"] = incidence_angle_list
            hirise_dataframe["EMISSION_ANGLE"] = emission_angle_list
            hirise_dataframe["PHASE_ANGLE"] = phase_angle_list
            hirise_dataframe["LOCAL_TIME"] = local_time_list
            hirise_dataframe["SOLAR_LONGITUDE"] = solar_longitude_list
            hirise_dataframe["SUB_SOLAR_AZIMUTH"] = sub_solar_azimuth_list
            hirise_dataframe["NORTH_AZIMUTH"] = north_azimuth_list

            hirise_dataframe["MRO:OBSERVATION_START_TIME"] = ob_start_time_list
            hirise_dataframe["START_TIME"] = start_time_list
            hirise_dataframe[
                "SPACECRAFT_CLOCK_START_COUNT"
            ] = spacecraft_start_list
            hirise_dataframe["STOP_TIME"] = product_creation_list
            hirise_dataframe[
                "SPACECRAFT_CLOCK_STOP_COUNT"
            ] = spacecraft_stop_list
            hirise_dataframe["PRODUCT_CREATION_TIME"] = product_creation_list

            hirise_dataframe["SCALING_FACTOR"] = scaling_factor_list
            hirise_dataframe["OFFSET"] = offset_list
            hirise_dataframe["CENTER_FILTER_WAVELENGTH"] = wavelength_list
            hirise_dataframe["IMG_URL"] = image_url_list[
                download_range_list[i]: download_range_list[i + 1]
            ]
            hirise_dataframe["LABEL_URL"] = label_url_list[
                download_range_list[i]: download_range_list[i + 1]
            ]

            if __package__ is None or __package__ == "":
                hirise_dataframe.to_csv(
                    "hirise_data.csv",
                    mode="a",
                    header=False,
                    encoding="utf-8",
                    index=False,
                )
            else:
                hirise_dataframe.to_csv(
                    folder_path,
                    mode="a",
                    encoding="utf-8",
                    header=False,
                    index=False,
                )

            hirise_dataframe = pd.DataFrame()

    def download(self, Hiriseimg_objs, folder_path, data_reload=True):
        """Function that downloads HIRISE image object entered by the user"""
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
        """Function that downloads random HIRISE images from the
        local or config database based on number of images entered by the
         user"""
        if local_database_path:
            hirise_df = pd.read_csv(local_database_path)
        else:
            hirise_df = pd.read_csv(CSV_FILE_PATH)

        random_samples = hirise_df.sample(n=image_count)

        # Filenames extracted from the queried dataframe
        f_names = [str(x) for x in random_samples["FILE_NAME"]]

        # Create HIRISE img object outputs using the hirise_img class
        Hirise_img_objects = [
            Hirise_Image.HiriseImage(f_name) for f_name in f_names
        ]

        ImageClient.download(
            self, Hiriseimg_objs=Hirise_img_objects, folder_path=fol_path
        )

    def reload_science_theme_database(
        self,
        science_themes,
        image_count=1,
        user_name="niranjana",
        password="B2bcTFp!5AtEAs5",
    ):
        """Function that updates or downloads the science theme database"""
        cols = ["FILE_NAME", "SCIENCE_THEME"]
        themes_dataframe = pd.DataFrame(columns=cols)
        hirise_df = pd.read_csv(CSV_FILE_PATH)
        if isinstance(science_themes, str):
            science_themes = [science_themes]
        elif all(
            isinstance(item, str) for item in science_themes
        ):  # check iterable for stringness of all items. Will raise TypeError
            # if item is not iterable
            pass
        else:
            raise TypeError

        # Define the urls needed for the downloads
        login_url = "https://www.uahirise.org/hiwish/login"
        payload = {"username": user_name, "password": password}

        for theme in science_themes:
            # Login to the website
            science_theme_list = {
                "Climate Change": "1",
                "Eolian Process": "2",
                "Fluvial Process": "3",
                "Future Exploration/Landing Sites": "4",
                "Geologic Contacts/Stratigraphy": "5",
                "Glacial/Periglacial Processes": "6",
                "Hydrothermal Processes": "7",
                "Impact Process": "8",
                "Landscape Evolution": "9",
                "Mass Wasting Processes": "10",
                "Polar Geology": "11",
                "Seasonal Processes": "12",
                "Sedimentary/Layering Processes": "13",
                "Rocks and Regolith": "14",
                "Composition and Photometry": "15",
                "Tectonic Processes": "16",
                "Volcanic Processes": "17",
                "Other": "18",
            }
            with requests.Session() as connection:

                connection.post(login_url, data=payload)

                post_url = f"https://www.uahirise.org/hiwish/search?cenLat=" \
                           f"0.0&latRange=0.0&cenLon=0.0&lonRange=0.0&text=&" \
                           f"word=on&sd=on&" \
                           f"theme1={science_theme_list.get(theme)}&" \
                           f"username=&size=100000"
                req2 = connection.get(post_url)
                soup = BeautifulSoup(req2.text, "html.parser")
                all_tables = soup.find_all("table")
                value_table = all_tables[3]

                rows = value_table.find_all("tr")
                rows = rows[1:]
                science_theme_list = []
                observation_list = []
                labels_list = []

                for i in range(len(rows)):
                    labels_list.append(rows[i].find_all("a"))

                titles_list = []
                for label in labels_list:
                    for i in range(len(label)):
                        titles_list.append(label[i]["title"])

                observations = [
                    x.split(" ")[2]
                    for x in titles_list
                    if "View observation" in x
                ]
                for obs in observations:
                    select_row = hirise_df[
                        hirise_df["FILE_NAME"].str.contains(obs)
                    ]
                    string_fname = select_row["FILE_NAME"].tolist()
                    if string_fname != []:
                        string_fname = string_fname[0]

                        observation_list.append(string_fname)
                        science_theme_list.append(theme)
                        # Add lists to dataframe
                themes_dataframe["FILE_NAME"] = observation_list
                themes_dataframe["SCIENCE_PHASE"] = science_theme_list

                if __package__ is None or __package__ == "":
                    themes_dataframe.to_csv(
                        "theme_data.csv",
                        mode="a",
                        header=False,
                        encoding="utf-8",
                        index=False,
                    )
                else:
                    themes_dataframe.to_csv(
                        "folder_path",
                        mode="a",
                        encoding="utf-8",
                        header=False,
                        index=False,
                    )
                themes_dataframe = pd.DataFrame()

    def filter_science_theme(
        self, science_theme, image_count=15, local_database_path=False
    ):
        """Function that returns HIRISE image object filtered
        based on  science theme entered by the user"""
        cols = ["FILE_NAME", "SCIENCE_THEME"]
        if local_database_path:
            theme_df = pd.read_csv(THEME_FILE_PATH, names=cols)
        else:
            theme_df = pd.read_csv(THEME_FILE_PATH, names=cols)

        selected_theme_df = theme_df[
            theme_df["SCIENCE_THEME"] == science_theme
        ]
        random_samples = selected_theme_df.sample(n=image_count)

        # Filenames extracted from the queried dataframe
        f_names = [str(x) for x in random_samples["FILE_NAME"]]

        # Create HIRISE img object outputs using the hirise_img class
        Hirise_img_objects = [
            Hirise_Image.HiriseImage(f_name) for f_name in f_names
        ]

        return Hirise_img_objects

    def filter_by_title(
        self,
        title_keywords,
        image_count=None,
        return_titles=False,
        user_name="niranjana",
        password="B2bcTFp!5AtEAs5",
    ):
        """Function that returns HIRISE image object filtered
        based on title of the image entered by the user"""
        login_url = "https://www.uahirise.org/hiwish/login"
        payload = {"username": user_name, "password": password}
        with requests.Session() as connection:
            connection.post(login_url, data=payload)
            try:
                CSV_FILE_PATH = "./HIRISE_api/hirise/hirise_data.csv"
                hirise_df = pd.read_csv(CSV_FILE_PATH)
            except BaseException:
                CSV_FILE_PATH = "../HIRISE_api/hirise/hirise_data.csv"
                hirise_df = pd.read_csv(CSV_FILE_PATH)

            post_url = f"https://www.uahirise.org/hiwish/search?" \
                       f"cenLat=0.0&latRange=0.0&cenLon=0.0&lonRange=0.0&" \
                       f"text={title_keywords}&word=on&sd=on&username=&size=" \
                       f"100000"
            req2 = connection.get(post_url)
            soup = BeautifulSoup(req2.text, "html.parser")
            all_tables = soup.find_all("table")
            value_table = all_tables[3]

            rows = value_table.find_all("tr")
            rows = rows[1:]
            observation_list = []
            labels_list = []
            description_list = []

            for i in range(len(rows)):
                component = rows[i].find_all("a")
                data = component[1]
                data = data.get_text()
                description_list.append(data)

            for i in range(len(rows)):
                labels_list.append(rows[i].find_all("a"))

            titles_list = []
            for label in labels_list:
                for i in range(len(label)):
                    titles_list.append(label[i]["title"])

            observations = [
                x.split(" ")[2] for x in titles_list if "View observation" in x
            ]
            for obs in observations:
                select_row = hirise_df[
                    hirise_df["FILE_NAME"].str.contains(obs)
                ]
                string_fname = select_row["FILE_NAME"].tolist()
                if string_fname:
                    string_fname = string_fname[0]

                    observation_list.append(string_fname)
                    # Add lists to dataframe
                    # Create HIRISE img object outputs using the hirise_img
                    # class
            if image_count:
                random_samples = random.sample(observation_list, image_count)
            Hirise_img_objects = [
                Hirise_Image.HiriseImage(f_name) for f_name in random_samples
            ]
            # Return to parent directory

            if return_titles:
                return Hirise_img_objects, description_list
            else:
                return Hirise_img_objects

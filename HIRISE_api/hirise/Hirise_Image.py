from bs4 import BeautifulSoup
from pprint import pprint
import pandas as pd
from tqdm import tqdm
import humanize
import pkg_resources
print(dir())

if __package__ is None or __package__ == "":
    # uses current directory visibility
    CSV_FILE_PATH = "../../HIRISE_api/hirise/hirise_data.csv"
else:
    CSV_FILE_PATH = pkg_resources.resource_filename("hirise", "hirise_data.csv")


class Hirise_Image:
    "Class that creates an HIRISE image object that has specific attributes including latitude longitude"

    def __init__(self, file_name):
        self.file_name = file_name

    def get_file_name(self):
        return self.file_name

    def get_mission_phase(self, local_database_path=None):

        if local_database_path:
            hirise_df = pd.read_csv(local_database_path)
        else:
            hirise_df = pd.read_csv(CSV_FILE_PATH)

        observation_values = hirise_df[hirise_df["FILE_NAME"] == self.file_name]
        return observation_values["MISSION_PHASE_NAME"].values[0]

    def get_orbit_number(self, local_database_path=None):

        if local_database_path:
            hirise_df = pd.read_csv(local_database_path)
        else:
            hirise_df = pd.read_csv(CSV_FILE_PATH)
        observation_values = hirise_df[hirise_df["FILE_NAME"] == self.file_name]
        return observation_values["ORBIT_NUMBER"].values[0]

    def get_orbital_range(self, local_database_path=None):

        if local_database_path:
            hirise_df = pd.read_csv(local_database_path)
        else:
            hirise_df = pd.read_csv(CSV_FILE_PATH)
        observation_values = hirise_df[hirise_df["FILE_NAME"] == self.file_name]
        return observation_values["ORBITAL_RANGE"].values[0]

    def get_center_latitude(self, local_database_path=None):

        if local_database_path:
            hirise_df = pd.read_csv(local_database_path)
        else:
            hirise_df = pd.read_csv(CSV_FILE_PATH)
        observation_values = hirise_df[hirise_df["FILE_NAME"] == self.file_name]
        return observation_values["CENTER_LATITUDE"].values[0]

    def get_center_longitude(self, local_database_path=None):

        if local_database_path:
            hirise_df = pd.read_csv(local_database_path)
        else:
            hirise_df = pd.read_csv(CSV_FILE_PATH)
        observation_values = hirise_df[hirise_df["FILE_NAME"] == self.file_name]
        return observation_values["CENTER_LONGITUDE"].values[0]

    def get_max_latitude(self, local_database_path=None):

        if local_database_path:
            hirise_df = pd.read_csv(local_database_path)
        else:
            hirise_df = pd.read_csv(CSV_FILE_PATH)
        observation_values = hirise_df[hirise_df["FILE_NAME"] == self.file_name]
        return observation_values["MAXIMUM_LATITUDE"].values[0]

    def get_min_latitude(self, local_database_path=None):

        if local_database_path:
            hirise_df = pd.read_csv(local_database_path)
        else:
            hirise_df = pd.read_csv(CSV_FILE_PATH)
        observation_values = hirise_df[hirise_df["FILE_NAME"] == self.file_name]
        return observation_values["MINIMUM_LATITUDE"].values[0]

    def get_easternmost_longitude(self, local_database_path=None):

        if local_database_path:
            hirise_df = pd.read_csv(local_database_path)
        else:
            hirise_df = pd.read_csv(CSV_FILE_PATH)
        observation_values = hirise_df[hirise_df["FILE_NAME"] == self.file_name]
        return observation_values["EASTERNMOST_LONGITUDE"].values[0]

    def get_westernmost_longitude(self, local_database_path=None):

        if local_database_path:
            hirise_df = pd.read_csv(local_database_path)
        else:
            hirise_df = pd.read_csv(CSV_FILE_PATH)
        observation_values = hirise_df[hirise_df["FILE_NAME"] == self.file_name]
        return observation_values["WESTERNMOST_LONGITUDE"].values[0]

    def get_incidence_angle(self, local_database_path=None):

        if local_database_path:
            hirise_df = pd.read_csv(local_database_path)
        else:
            hirise_df = pd.read_csv(CSV_FILE_PATH)
        observation_values = hirise_df[hirise_df["FILE_NAME"] == self.file_name]
        return observation_values["INCIDENCE_ANGLE"].values[0]

    def get_emission_angle(self, local_database_path=None):

        if local_database_path:
            hirise_df = pd.read_csv(local_database_path)
        else:
            hirise_df = pd.read_csv(CSV_FILE_PATH)
        observation_values = hirise_df[hirise_df["FILE_NAME"] == self.file_name]
        return observation_values["EMISSION_ANGLE"].values[0]

    def get_phase_angle(self, local_database_path=None):

        if local_database_path:
            hirise_df = pd.read_csv(local_database_path)
        else:
            hirise_df = pd.read_csv(CSV_FILE_PATH)
        observation_values = hirise_df[hirise_df["FILE_NAME"] == self.file_name]
        return observation_values["PHASE_ANGLE"].values[0]

    def get_local_time(self, local_database_path=None):

        if local_database_path:
            hirise_df = pd.read_csv(local_database_path)
        else:
            hirise_df = pd.read_csv(CSV_FILE_PATH)
        observation_values = hirise_df[hirise_df["FILE_NAME"] == self.file_name]
        return observation_values["LOCAL_TIME"].values[0]

    def get_solar_longitude(self, local_database_path=None):

        if local_database_path:
            hirise_df = pd.read_csv(local_database_path)
        else:
            hirise_df = pd.read_csv(CSV_FILE_PATH)
        observation_values = hirise_df[hirise_df["FILE_NAME"] == self.file_name]
        return observation_values["SOLAR_LONGITUDE"].values[0]

    def get_sub_solar_azimuth(self, local_database_path=None):

        if local_database_path:
            hirise_df = pd.read_csv(local_database_path)
        else:
            hirise_df = pd.read_csv(CSV_FILE_PATH)
        observation_values = hirise_df[hirise_df["FILE_NAME"] == self.file_name]
        return observation_values["SUB_SOLAR_AZIMUTH"].values[0]

    def get_north_azimuth(self, local_database_path=None):

        if local_database_path:
            hirise_df = pd.read_csv(local_database_path)
        else:
            hirise_df = pd.read_csv(CSV_FILE_PATH)
        observation_values = hirise_df[hirise_df["FILE_NAME"] == self.file_name]
        return observation_values["NORTH_AZIMUTH"].values[0]

    def get_mro_observation_start_time(self, local_database_path=None):

        if local_database_path:
            hirise_df = pd.read_csv(local_database_path)
        else:
            hirise_df = pd.read_csv(CSV_FILE_PATH)
        observation_values = hirise_df[hirise_df["FILE_NAME"] == self.file_name]
        return observation_values["MRO:OBSERVATION_START_TIME"].values[0]

    def get_start_time(self, local_database_path=None):

        if local_database_path:
            hirise_df = pd.read_csv(local_database_path)
        else:
            hirise_df = pd.read_csv(CSV_FILE_PATH)
        observation_values = hirise_df[hirise_df["FILE_NAME"] == self.file_name]
        return observation_values["START_TIME"].values[0]

    def get_spacecraft_clock_start_count(self, local_database_path=None):

        if local_database_path:
            hirise_df = pd.read_csv(local_database_path)
        else:
            hirise_df = pd.read_csv(CSV_FILE_PATH)
        observation_values = hirise_df[hirise_df["FILE_NAME"] == self.file_name]
        return observation_values["SPACECRAFT_CLOCK_START_COUNT"].values[0]

    def get_stop_time(self, local_database_path=None):

        if local_database_path:
            hirise_df = pd.read_csv(local_database_path)
        else:
            hirise_df = pd.read_csv(CSV_FILE_PATH)
        observation_values = hirise_df[hirise_df["FILE_NAME"] == self.file_name]
        return observation_values["STOP_TIME"].values[0]

    def get_spacecraft_clock_stop_time(self, local_database_path=None):

        if local_database_path:
            hirise_df = pd.read_csv(local_database_path)
        else:
            hirise_df = pd.read_csv(CSV_FILE_PATH)
        observation_values = hirise_df[hirise_df["FILE_NAME"] == self.file_name]
        return observation_values["SPACECRAFT_CLOCK_STOP_COUNT"].values[0]

    def get_product_creation_time(self, local_database_path=None):

        if local_database_path:
            hirise_df = pd.read_csv(local_database_path)
        else:
            hirise_df = pd.read_csv(CSV_FILE_PATH)
        observation_values = hirise_df[hirise_df["FILE_NAME"] == self.file_name]
        return observation_values["PRODUCT_CREATION_TIME"].values[0]

    def get_scaling_factor(self, local_database_path=None):

        if local_database_path:
            hirise_df = pd.read_csv(local_database_path)
        else:
            hirise_df = pd.read_csv(CSV_FILE_PATH)
        observation_values = hirise_df[hirise_df["FILE_NAME"] == self.file_name]
        return observation_values["SCALING_FACTOR"].values[0]

    def get_offset(self, local_database_path=None):

        if local_database_path:
            hirise_df = pd.read_csv(local_database_path)
        else:
            hirise_df = pd.read_csv(CSV_FILE_PATH)
        observation_values = hirise_df[hirise_df["FILE_NAME"] == self.file_name]
        return observation_values["OFFSET"].values[0]

    def get_center_filter_wavelength(self, local_database_path=None):

        if local_database_path:
            hirise_df = pd.read_csv(local_database_path)
        else:
            hirise_df = pd.read_csv(CSV_FILE_PATH)
        observation_values = hirise_df[hirise_df["FILE_NAME"] == self.file_name]
        return observation_values["CENTER_FILTER_WAVELENGTH"].values[0]

    def get_img_url(self, local_database_path=None):

        if local_database_path:
            hirise_df = pd.read_csv(local_database_path)
        else:
            hirise_df = pd.read_csv(CSV_FILE_PATH)
        observation_values = hirise_df[hirise_df["FILE_NAME"] == self.file_name]
        return observation_values["IMG_URL"].values[0]

    def get_label_url(self, local_database_path=None):

        if local_database_path:
            hirise_df = pd.read_csv(local_database_path)
        else:
            hirise_df = pd.read_csv(CSV_FILE_PATH)
        observation_values = hirise_df[hirise_df["FILE_NAME"] == self.file_name]
        return str(observation_values["LABEL_URL"].values[0])

    def get_image_size(self, local_database_path=None):

        if local_database_path:
            hirise_df = pd.read_csv(local_database_path)
        else:
            hirise_df = pd.read_csv(CSV_FILE_PATH)
        observation_values = hirise_df[hirise_df["FILE_NAME"] == self.file_name]
        return observation_values["REQUIRED_STORAGE_BYTES"].values[0]

    def get_file_parameters(self, local_database_path=None):

        if local_database_path:
            hirise_df = pd.read_csv(local_database_path)
        else:
            hirise_df = pd.read_csv(CSV_FILE_PATH)
        observation_values = hirise_df[hirise_df["FILE_NAME"] == self.file_name]
        return {
            "FILE_NAME": self.file_name,
            "IMG_SIZE": humanize.naturalsize(
                observation_values["REQUIRED_STORAGE_BYTES"]
            ),
            "REQUIRED_STORAGE_BYTES": observation_values[
                "REQUIRED_STORAGE_BYTES"
            ].values[0],
            "MISSION_PHASE_NAME": observation_values["MISSION_PHASE_NAME"].values[0],
            "ORBIT_NUMBER": observation_values["ORBIT_NUMBER"].values[0],
            "ORBITAL_RANGE": observation_values["ORBITAL_RANGE"].values[0],
        }

    def get_viewing_parameters(self, local_database_path=None):

        if local_database_path:
            hirise_df = pd.read_csv(local_database_path)
        else:
            hirise_df = pd.read_csv(CSV_FILE_PATH)
        observation_values = hirise_df[hirise_df["FILE_NAME"] == self.file_name]
        return {
            "INCIDENCE_ANGLE": observation_values["INCIDENCE_ANGLE"].values[0],
            "EMISSION_ANGLE": observation_values["EMISSION_ANGLE"].values[0],
            "PHASE_ANGLE": observation_values["PHASE_ANGLE"].values[0],
            "LOCAL_TIME": observation_values["LOCAL_TIME"].values[0],
            "SOLAR_LONGITUDE": observation_values["SOLAR_LONGITUDE"].values[0],
            "SUB_SOLAR_AZIMUTH": observation_values["SUB_SOLAR_AZIMUTH"].values[0],
            "NORTH_AZIMUTH": observation_values["NORTH_AZIMUTH"].values[0],
        }

    def get_timing_parameters(self, local_database_path=None):

        if local_database_path:
            hirise_df = pd.read_csv(local_database_path)
        else:
            hirise_df = pd.read_csv(CSV_FILE_PATH)
        observation_values = hirise_df[hirise_df["FILE_NAME"] == self.file_name]
        return {
            "MRO:OBSERVATION_START_TIME": observation_values[
                "MRO:OBSERVATION_START_TIME"
            ].values[0],
            "START_TIME": observation_values["START_TIME"].values[0],
            "SPACECRAFT_CLOCK_START_COUNT": observation_values[
                "SPACECRAFT_CLOCK_START_COUNT"
            ].values[0],
            "STOP_TIME": observation_values["STOP_TIME"].values[0],
            "SPACECRAFT_CLOCK_STOP_COUNT": observation_values[
                "SPACECRAFT_CLOCK_STOP_COUNT"
            ].values[0],
            "PRODUCT_CREATION_TIME": observation_values["PRODUCT_CREATION_TIME"].values[
                0
            ],
        }

    def get_other_parameters(self, local_database_path=None):

        if local_database_path:
            hirise_df = pd.read_csv(local_database_path)
        else:
            hirise_df = pd.read_csv(CSV_FILE_PATH)
        observation_values = hirise_df[hirise_df["FILE_NAME"] == self.file_name]
        return {
            "SCALING_FACTOR": observation_values["SCALING_FACTOR"].values[0],
            "OFFSET": observation_values["OFFSET"].values[0],
            "CENTER_FILTER_WAVELENGTH": observation_values[
                "CENTER_FILTER_WAVELENGTH"
            ].values[0],
        }

    def get_all_parameters(self, local_database_path=None):
        """Function that returns all the"""
        if local_database_path:
            hirise_df = pd.read_csv(local_database_path)
        else:
            hirise_df = pd.read_csv(CSV_FILE_PATH)

        observation_values = hirise_df[hirise_df["FILE_NAME"] == self.file_name]
        return {
            "FILE_NAME": self.file_name,
            "IMG_SIZE": humanize.naturalsize(
                int(observation_values["REQUIRED_STORAGE_BYTES"].values[0])
            ),
            "REQUIRED_STORAGE_BYTES": observation_values[
                "REQUIRED_STORAGE_BYTES"
            ].values[0],
            "MISSION_PHASE_NAME": observation_values["MISSION_PHASE_NAME"].values[0],
            "ORBIT_NUMBER": observation_values["ORBIT_NUMBER"].values[0],
            "ORBITAL_RANGE": observation_values["ORBITAL_RANGE"].values[0],
            "CENTER_LATITUDE": observation_values["CENTER_LATITUDE"].values[0],
            "CENTER_LONGITUDE": observation_values["CENTER_LONGITUDE"].values[0],
            "MAXIMUM_LATITUDE": observation_values["MAXIMUM_LATITUDE"].values[0],
            "MINIMUM_LATITUDE": observation_values["MINIMUM_LATITUDE"].values[0],
            "EASTERNMOST_LONGITUDE": observation_values["EASTERNMOST_LONGITUDE"].values[
                0
            ],
            "WESTERNMOST_LONGITUDE": observation_values["WESTERNMOST_LONGITUDE"].values[
                0
            ],
            "INCIDENCE_ANGLE": observation_values["INCIDENCE_ANGLE"].values[0],
            "EMISSION_ANGLE": observation_values["EMISSION_ANGLE"].values[0],
            "PHASE_ANGLE": observation_values["PHASE_ANGLE"].values[0],
            "LOCAL_TIME": observation_values["LOCAL_TIME"].values[0],
            "SOLAR_LONGITUDE": observation_values["SOLAR_LONGITUDE"].values[0],
            "SUB_SOLAR_AZIMUTH": observation_values["SUB_SOLAR_AZIMUTH"].values[0],
            "NORTH_AZIMUTH": observation_values["NORTH_AZIMUTH"].values[0],
            "MRO:OBSERVATION_START_TIME": observation_values[
                "MRO:OBSERVATION_START_TIME"
            ].values[0],
            "START_TIME": observation_values["START_TIME"].values[0],
            "SPACECRAFT_CLOCK_START_COUNT": observation_values[
                "SPACECRAFT_CLOCK_START_COUNT"
            ].values[0],
            "STOP_TIME": observation_values["STOP_TIME"].values[0],
            "SPACECRAFT_CLOCK_STOP_COUNT": observation_values[
                "SPACECRAFT_CLOCK_STOP_COUNT"
            ].values[0],
            "PRODUCT_CREATION_TIME": observation_values["PRODUCT_CREATION_TIME"].values[
                0
            ],
            "SCALING_FACTOR": observation_values["SCALING_FACTOR"].values[0],
            "OFFSET": observation_values["OFFSET"].values[0],
            "CENTER_FILTER_WAVELENGTH": observation_values[
                "CENTER_FILTER_WAVELENGTH"
            ].values[0],
            "IMG_URL": observation_values["IMG_URL"].values[0],
            "LABEL_URL": observation_values["LABEL_URL"].values[0],
        }

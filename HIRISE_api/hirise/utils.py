import requests
from bs4 import BeautifulSoup


def file_parameters_list():
    """
    Function that returns file parameters list
    """
    return [
        "FILE_NAME",
        "REQUIRED_STORAGE_BYTES",
        "MISSION_PHASE_NAME",
        "ORBIT_NUMBER",
    ]


def image_map_parameters_list():
    """
    Function that returns image mapping parameters list
    """
    return [
        "CENTER_LATITUDE",
        "CENTER_LONGITUDE",
        "MAXIMUM_LATITUDE",
        "MINIMUM_LATITUDE",
        "EASTERNMOST_LONGITUDE",
        "WESTERNMOST_LONGITUDE",
    ]


def viewing_parameters_list():
    """
    Function that returns viewing parameters list
    """
    return [
        "INCIDENCE_ANGLE",
        "EMISSION_ANGLE",
        "PHASE_ANGLE",
        "LOCAL_TIME",
        "SOLAR_LONGITUDE",
        "SUB_SOLAR_AZIMUTH",
        "NORTH_AZIMUTH",
    ]


def timing_parameters_list():
    """
    Function that returns timing parameters list
    """
    return [
        "MRO:OBSERVATION_START_TIME",
        "START_TIME",
        "SPACECRAFT_CLOCK_START_COUNT",
        "STOP_TIME",
        "SPACECRAFT_CLOCK_STOP_COUNT",
        "PRODUCT_CREATION_TIME",
    ]


def other_parameters_list():
    """
    Function that returns scaling factor, offset, center filter wavelength
    parameters list
    """
    return ["SCALING_FACTOR", "OFFSET", "CENTER_FILTER_WAVELENGTH"]


def LBL_parser(label_url):
    """
    Function that parses the .LBL file in NASA's Planetary Data System
    """
    # Get the data from the url
    req = requests.get(label_url, stream=True)

    lbl_file = req.text
    lbl_dict = {
        "file_parameters": {},
        "image_map_parameters": {},
        "viewing_parameters": {},
        "timing_parameters": {},
        "other_parameters": {},
    }

    # Select the parameters to save
    file_parameters = file_parameters_list()
    image_map_parameters = image_map_parameters_list()
    viewing_parameters = viewing_parameters_list()
    timing_parameters = timing_parameters_list()
    other_parameters = other_parameters_list()

    file_dict = dict.fromkeys(file_parameters, "")
    image_dict = dict.fromkeys(image_map_parameters, "")
    view_dict = dict.fromkeys(viewing_parameters, "")
    time_dict = dict.fromkeys(timing_parameters, "")
    other_dict = dict.fromkeys(other_parameters, "")

    for line in lbl_file.split("\n"):
        # Split the label data
        if "=" in line:
            line_array = line.split("=")
            line_key = line_array[0].strip()
            line_value = line_array[1].strip()

            # Append required values in the respective df formats
            if line_key in file_parameters:
                file_dict[line_key] = line_value
            elif line_key in viewing_parameters:
                view_dict[line_key] = line_value
            elif line_key in timing_parameters:
                time_dict[line_key] = line_value
            elif line_key in other_parameters:
                other_dict[line_key] = line_value
            elif line_key in image_map_parameters:
                image_dict[line_key] = line_value

    lbl_dict["file_parameters"] = file_dict
    lbl_dict["image_map_parameters"] = image_dict
    lbl_dict["viewing_parameters"] = view_dict
    lbl_dict["timing_parameters"] = time_dict
    lbl_dict["other_parameters"] = other_dict

    return lbl_dict


def get_website_data(base_url, page_key, sub_key=None):
    """
    Function that assistes in wescaping the NASA website
    """
    if sub_key:
        page_url = base_url + str(page_key) + "/" + sub_key
    else:
        page_url = base_url + str(page_key)

    # Get the data from the url
    req = requests.get(page_url)

    # Use soup to create an HTML parser
    soup = BeautifulSoup(req.text, "html.parser")

    # Find all the labels with the correct tag - "a"
    labels = soup.find_all("a")
    # List to save orbital labels
    page_labels = []

    # Loop through all the orbital ranges
    for i in range(1, len(labels)):
        # Save orbital labels into a list
        page_labels.append(labels[i]["href"])

    return page_labels, len(page_labels)


def validate_append_float_data(param, list_of_params):
    """
    Function that validates floating point data
    """
    if param:
        list_of_params.append(float(param.split("<")[0]))
    else:
        list_of_params.append(None)


def append_float_data_without_strip(param, list_of_params):
    """
    Function that validates floating point data without stripping the last
    characters
    """
    if param:
        list_of_params.append(float(param))
    else:
        list_of_params.append(None)


def downloadRange(start_range, end, step):
    i = start_range
    while i < end:
        yield i
        i += step
    yield end

# from . import hirise
# from .. import hirise,preprocessing,models
from HIRISE_api.hirise import Image_Client, Hirise_Image
import os.path
import pkg_resources


if __package__ is None or __package__ == "":
    # uses current directory visibility
    CSV_FILE_PATH = "./HIRISE_api/hirise/hirise_data.csv"
    THEME_FILE_PATH = "./HIRISE_api/hirise/theme_data.csv"
else:
    CSV_FILE_PATH = pkg_resources.resource_filename("hirise", "hirise_data.csv")
    THEME_FILE_PATH = pkg_resources.resource_filename("hirise", "theme_data.csv")



def test_get_all_parameters():
    hirise_image = Hirise_Image.HiriseImage('PSP_008585_2915_COLOR.IMG')
    image_parameters = hirise_image.get_all_parameters()
    assert(image_parameters['FILE_NAME'] == 'PSP_008585_2915_COLOR.IMG')
    assert (image_parameters['MISSION_PHASE_NAME'] == 'PRIMARY SCIENCE PHASE')

def test_get_individual_parameters():
    hirise_image_object = Hirise_Image("PSP_008585_2915_COLOR.IMG")
    orbit_number = hirise_image_object.get_orbit_number()
    scaling_factor = hirise_image_object.get_scaling_factor()
    assert(orbit_number == 8585)
    assert(scaling_factor == 3.67e-05)


def test_database_exists():
    assert(os.path.exists(CSV_FILE_PATH))
    assert(os.path.exists(THEME_FILE_PATH))



def test_get_images():
    hirise_image_client = Image_Client.ImageClient()
    images_result = hirise_image_client.get_images(
        file_name='PSP_006281_0965_COLOR.IMG'
    )

    assert images_result


def test_filter_center_latlon():
    hirise_image_client = Image_Client.ImageClient()
    images_result = hirise_image_client.filter_center_latlon(
        center_latlon=[90,0]
    )
    assert images_result


def test_filter_by_title():
    hirise_image_client = Image_Client.ImageClient()
    images_result = hirise_image_client.filter_by_title(
        title_keywords='Hidden Water Channel', image_count=1
    )
    assert images_result

def test_filter_by_mission_phase():
    hirise_image_client = Image_Client.ImageClient()
    images_result = hirise_image_client.filter_mission_phase(
        mission_phase='PRIMARY SCIENCE PHASE', image_count=1
    )
    assert images_result



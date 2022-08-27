from hirise.Image_Client import ImageClient


def test_get_images():
    hirise_image_client = ImageClient()
    images_result = hirise_image_client.get_images(
        file_name='PSP_006281_0965_COLOR.IMG'
    )

    assert images_result


def test_filter_center_latlon():
    hirise_image_client = ImageClient()
    images_result = hirise_image_client.filter_center_latlon(
        center_latlon=0
    )
    assert images_result


def test_filter_by_title():
    hirise_image_client = ImageClient()
    images_result = hirise_image_client.filter_by_title(
        title_keywords='Hidden Water Channel'
    )
    assert images_result


def test_filter_science_theme():
    hirise_image_client = ImageClient()
    images_result = hirise_image_client.filter_science_theme(
        science_theme = 'PSP', image_count= 1
    )
    assert images_result

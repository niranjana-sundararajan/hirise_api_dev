from hirise.Hirise_Image import HiriseImage

def test_get_all_parameters():
    hirise_image = HiriseImage('PSP_008585_2915_COLOR.IMG')
    image_parameters = hirise_image.get_all_parameters()
    print(image_parameters)
    assert(image_parameters['FILE_NAME'] == 'PSP_008585_2915_COLOR.IMG')
    assert (image_parameters['MISSION_PHASE_NAME'] == 'PRIMARY SCIENCE PHASE')

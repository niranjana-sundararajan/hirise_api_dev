import pytest
import os, sys
# import shutil
from pathlib import Path

# Set path to parent directory 
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)


from hirise import download


def test_downloads():   
    folder_pth = '..\data-download'
    download.download_files(["ESP"],orb_range=[11200,11299], download_data=False, folder_path=folder_pth)
    dir = os.listdir(folder_pth)

    assert len(dir)!= 0

  
sys.path.insert(0, parent_dir_path)
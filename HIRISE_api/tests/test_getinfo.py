import os, sys
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
import pytest
from hirise import info
from pprint import pprint
def test_get_info():
    # TEST 1 
    # Check if data and paths exist for each mission
    mission_phases = ["AEB", 'TRA']
    for mission_phase in mission_phases:
      data = info.get_info(mission_phase, calculate_size=False)
     # Check that data is not null
      assert data != []

def test_get_info_filters():    
    # TEST 2 
    # Check if filtered information work   
    mission_1 = 'ESP'
    orbital_range = [11200,11499]
    data = info.get_info(mission_1, orbital_range, calculate_size=False)
    assert data != []


import sys
import os

sys.path.insert(0, os.path.abspath(os.sep.join((os.curdir, '..'))))

project = 'HIRISE_api'
author = ' '
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              'sphinx.ext.autosummary']
source_suffix = '.rst'
master_doc = 'index'
exclude_patterns = ['_build']
autoclass_content = "both"
html_theme_options = {'navigation_depth': 2}

autosummary_generate = True
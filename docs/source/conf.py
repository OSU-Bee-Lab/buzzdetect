import sys
import os
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------

project = 'buzzdetect'
copyright = '2025, OSU Bee Lab'
author = 'OSU Bee Lab'

# The short X.Y version
version = '1.0'

# The full version, including alpha/beta/rc tags
release = '1.0.1'


# -- General configuration ---------------------------------------------------
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'sphinx_rtd_theme'

extensions = [
    'sphinx.ext.napoleon',  # for Google/Numpy style docstrings
    'sphinx_rtd_theme',
    'sphinx.ext.autodoc',  # generate API docs from docstrings
    'sphinx.ext.autosummary'  # recursively search and document module
]

# autodoc
autodoc_member_order = 'groupwise'  # Group members by type
autoclass_content = 'both'  # Include both class and __init__ docstrings

# autosummary
autosummary_generate = True  # Turn on sphinx.ext.autosummary
templates_path = ['_templates']
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'special-members': '__init__',
    # Optional, if useful:
    # 'inherited-members': True,
    # 'show-inheritance': True,
}
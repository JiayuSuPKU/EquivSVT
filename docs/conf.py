# Configuration file for the Sphinx documentation builder.
# For full options, see the Sphinx documentation at https://www.sphinx-doc.org/en/master/config

import os
import sys

# Add quadsv source directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Project information
project = 'quadsv'
copyright = '2026, Jiayu Su'
author = 'Jiayu Su'
version = '0.1.0'
release = '0.1.0'

# General configuration
extensions = [
    'sphinx.ext.autodoc',        # Auto-generate API docs from docstrings
    'sphinx.ext.napoleon',        # NumPy-style docstring parsing
    'sphinx.ext.mathjax',         # Math equation rendering
    'sphinx.ext.viewcode',        # Link to source code
    'sphinx.ext.intersphinx',     # Cross-reference other projects
    'myst_parser',                # Markdown support (.md files)
]

# Source file patterns
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# HTML theme
html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    "navbar_align": "left",
    "logo": {
        "text": "quadsv",
    },
    "search_bar_text": "Search documentation...",
    "show_toc_level": 2,
    "navigation_depth": 3,
    "show_nav_level": 2,
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
}

# Sidebar options
html_sidebars = {
    "**": ["search-field", "sidebar-nav-bs", "page-toc"]
}

# HTML output options
html_output_path = '_build/html'
html_static_path = ['_static']

# Napoleon (Google/NumPy docstring) options
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True

# Autodoc options
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': False,
    'show-inheritance': True,
}

# Intersphinx mapping (for cross-references)
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
}

# LaTeX options for math rendering (MathJax 4)
mathjax4_config = {
    'tex': {
        'inlinemath': [['$', '$'], ['\\(', '\\)']],
        'displaymath': [['$$', '$$'], ['\\[', '\\]']],
    }
}

# Try to generate plots with matplotlib
plot_gallery = False
plot_html_show_source_link = False

# Suppress certain warnings
suppress_warnings = ['ref.citation']

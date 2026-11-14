"""Sphinx configuration of the documentation of sklearn-contrib-emeans."""

# SPDX-License-Identifier: BSD-3-Clause

from importlib.metadata import version as get_version

project = "sklearn-contrib-emeans"
copyright = "2026, The sklearn-contrib-emeans developers"
release = get_version("sklearn-contrib-emeans")
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    # Adds the .nojekyll file that GitHub Pages needs to serve _static.
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "numpydoc",
    "sphinx_gallery.gen_gallery",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "_templates"]

# `text` is rendered as code, as in the docstrings of scikit-learn.
default_role = "literal"

# -- HTML output -------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = project
html_favicon = "_static/favicon.svg"
html_theme_options = {
    "github_url": "https://github.com/cakedev0/sklearn-contrib-emeans",
    "show_toc_level": 2,
    "show_prev_next": False,
}
# The site is a few pages: no section navigation on the left.
html_sidebars = {"**": []}

# -- autodoc, autosummary and numpydoc ---------------------------------------

autodoc_default_options = {"members": True, "inherited-members": True}
autosummary_generate = True
# The methods are documented by autodoc on the page of the class.
numpydoc_show_class_members = False

# -- intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

# -- sphinx-gallery ----------------------------------------------------------

sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    # Where the examples using each object are listed, for the API reference.
    "backreferences_dir": "generated",
    "doc_module": ("sklearn_emeans",),
}

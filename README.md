ipysliceviewer
===============================

An IPyWidget for viewing cross-sections of objects

Installation
------------

To install use pip:

    $ pip install ipysliceviewer
    $ jupyter nbextension enable --py --sys-prefix ipysliceviewer

To install for jupyterlab

    $ jupyter labextension install ipysliceviewer

For a development installation (requires npm),

    $ git clone https://github.com/tslaton/ipysliceviewer.git
    $ cd ipysliceviewer
    $ pip install -e .
    $ jupyter nbextension install --py --symlink --sys-prefix ipysliceviewer
    $ jupyter nbextension enable --py --sys-prefix ipysliceviewer

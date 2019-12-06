from ._version import version_info, __version__

from .sliceviewer import *

def _jupyter_nbextension_paths():
    return [{
        'section': 'notebook',
        'src': 'static',
        'dest': 'ipysliceviewer',
        'require': 'ipysliceviewer/extension'
    }]

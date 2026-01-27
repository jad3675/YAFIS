# UI package initialization

# Import main window and panels to make them easily accessible
from .main_window import MainWindow
from .adjustment_panel import AdjustmentPanel
from .preset_panel import FilmPresetPanel
from .photo_preset_panel import PhotoPresetPanel
from .image_viewer import ImageViewer
from .curves_widget import CurvesWidget

# You can define __all__ if you want to control what 'from .ui import *' imports
# __all__ = ['MainWindow', 'AdjustmentPanel', 'FilmPresetPanel', 'PhotoPresetPanel', 'ImageViewer', 'CurvesWidget']
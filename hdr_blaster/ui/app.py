"""Starting the window."""
import os
import sys

from PySide6.QtCore import QCoreApplication, QLocale, QStandardPaths, Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from .. import configure_logging
from . import theme


APP_ID = "HDRBlaster.HDRBlaster"


def app_icon() -> QIcon:
    """The icon of the window and of the taskbar. The files sit next to this module, in the
    source tree and inside the frozen executable alike."""
    here = os.path.dirname(os.path.abspath(__file__))
    icon = QIcon(os.path.join(here, "icon.ico"))
    if not icon.availableSizes():                      # no ICO reader in this Qt: the PNG will do
        icon = QIcon(os.path.join(here, "icon.png"))
    return icon


def create_application(argv=None) -> QApplication:
    if sys.platform == "win32":
        # Windows groups taskbar buttons by this identity. Without one, a program run by the
        # Python interpreter is filed under Python and shown with a blank icon.
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
        except Exception:
            pass
    # The HDR surface is a native window inside the picture area: it must not turn every
    # other widget of the window into a native window too.
    QCoreApplication.setAttribute(Qt.AA_DontCreateNativeWidgetSiblings)
    app = QApplication.instance() or QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("HDR Blaster")
    from .. import __version__

    app.setApplicationVersion(__version__)
    app.setWindowIcon(app_icon())
    QLocale.setDefault(QLocale(QLocale.English, QLocale.UnitedStates))   # the interface is in English: 1.00, not 1,00
    app.setStyle("Fusion")                 # the same widgets on every Windows version; the style sheet does the rest
    app.setFont(theme.app_font())
    app.setStyleSheet(theme.STYLE)
    return app


def main(argv=None, settings_path: str = "") -> int:
    """Open the window. Files given on the command line are queued. ``settings_path``: a
    file to keep the settings in, instead of the user's own (the tests give one)."""
    argv = list(sys.argv if argv is None else argv)
    # PyTorch takes seconds to import: a thread starts on it now, and the window opens
    # without waiting. The background thread of the window waits for it before it computes.
    from .workers import start_loading_engine

    start_loading_engine()
    app = create_application(argv)
    # The log goes to the user's own application folder: the current folder may be
    # anywhere, and not writable, when the program is started by a double-click.
    folder = QStandardPaths.writableLocation(QStandardPaths.AppLocalDataLocation)
    try:
        os.makedirs(folder, exist_ok=True)
        configure_logging(os.path.join(folder, "hdr_blaster.log"))
    except OSError:
        configure_logging(os.devnull)
    from .window import MainWindow

    window = MainWindow(settings_path)
    window.show()
    paths = [a for a in argv[1:] if not a.startswith("-")]
    if paths:
        window.add_paths(paths)
    return app.exec()

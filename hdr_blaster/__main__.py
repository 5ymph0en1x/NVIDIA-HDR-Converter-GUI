"""``python -m hdr_blaster`` opens the window; with --convert-video or --convert it runs headless,
and --version says which version this is."""
import sys


def run() -> int:
    if "--version" in sys.argv:
        from . import __version__
        print(f"HDR Blaster (NVIDIA HDR Converter GUI) {__version__}")
        return 0
    if "--convert-video" in sys.argv:
        from .cli import _cli_convert_video
        return _cli_convert_video(sys.argv[1:])
    if "--selftest" in sys.argv or "--convert" in sys.argv:
        from .cli import _cli_convert
        return _cli_convert(sys.argv[1:])
    from .ui.app import main
    return main()


if __name__ == "__main__":
    sys.exit(run())

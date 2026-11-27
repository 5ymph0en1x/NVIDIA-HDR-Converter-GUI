"""HDR Blaster engine: HDR captures to SDR, tone mapping and enhancement.

Importing the package has no side effect. The command line calls
:func:`configure_runtime` once, before it computes anything. The window calls its two
halves: :func:`configure_logging` at once, and :func:`configure_torch` from the thread that
loads PyTorch in the background, so that the window does not wait for it.
"""
import logging

__version__ = "3.0.3"

_logging_configured = False
_torch_configured = False


def configure_runtime(log_file: str = 'hdr_converter.log') -> None:
    """Set up logging and deterministic torch, exactly as the original NHC.py did at import."""
    configure_logging(log_file)
    configure_torch()


def configure_logging(log_file: str = 'hdr_converter.log') -> None:
    """The log file and the console. Needs nothing heavy: done before the window opens."""
    global _logging_configured
    if _logging_configured:
        return
    _logging_configured = True

    logging.basicConfig(
        filename=log_file,
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logging.getLogger('').addHandler(console)


def configure_torch() -> None:
    """Deterministic torch, before anything is computed. Importing PyTorch takes seconds:
    whoever calls this first pays for it."""
    global _torch_configured
    if _torch_configured:
        return
    _torch_configured = True

    import torch

    # Reproducibility: same settings and seed as the original module-level code.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

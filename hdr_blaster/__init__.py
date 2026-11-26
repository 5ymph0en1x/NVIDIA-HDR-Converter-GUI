"""HDR Blaster engine: HDR captures to SDR, tone mapping and enhancement.

Importing the package has no side effect. Entry points (the GUI, the CLI) call
:func:`configure_runtime` once, before they import the engine modules.
"""
import logging

__version__ = "3.0.0"

_configured = False


def configure_runtime(log_file: str = 'hdr_converter.log') -> None:
    """Set up logging and deterministic torch, exactly as the original NHC.py did at import."""
    global _configured
    if _configured:
        return
    _configured = True

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

    import torch

    # Reproducibility: same settings and seed as the original module-level code.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

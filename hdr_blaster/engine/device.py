"""GPU / CPU selection (moved verbatim from NHC.py)."""
import logging

import torch


class DeviceManager:
    """
    Manages the computational device (GPU or CPU) for the application.
    """

    def __init__(self):
        self.use_gpu = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        if self.use_gpu:
            torch.cuda.empty_cache()
        logging.info(f"DeviceManager initialized on {self.device}")

    def switch_device(self, use_gpu: bool):
        """
        Switches the device between GPU and CPU based on availability and user preference.
        """
        self.use_gpu = use_gpu and torch.cuda.is_available()
        new_device = torch.device("cuda" if self.use_gpu else "cpu")
        if new_device != self.device:
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            self.device = new_device
            logging.info(f"DeviceManager switched to {self.device}")

    def get_device(self):
        """Returns the current computational device."""
        return self.device

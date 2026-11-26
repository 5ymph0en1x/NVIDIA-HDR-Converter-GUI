"""HDR metadata read from JXR files (moved verbatim from NHC.py)."""
import logging
import struct
from typing import Dict


class HDRMetadata:
    """Stores and manages HDR metadata extracted from JXR files."""

    def __init__(self):
        self.max_luminance = 10000.0
        self.min_luminance = 0.0
        self.white_point = 1000.0
        self.color_primaries = "bt2020"
        self.transfer_function = "pq"
        self.color_space = "rec2020"
        self.bit_depth = 10
        self.has_metadata = False

    def extract_from_jxr(self, jxr_data: bytes) -> bool:
        """Extract HDR metadata from JXR file data."""
        try:
            # Look for HDR metadata markers in JXR container
            # This is a simplified version - real implementation would parse JPEGXR container properly
            if b'hdrf' in jxr_data:
                # Extract luminance values
                idx = jxr_data.find(b'hdrf')
                if idx != -1 and idx + 16 <= len(jxr_data):
                    # Parse metadata block (simplified)
                    try:
                        self.max_luminance = struct.unpack('<f', jxr_data[idx + 4:idx + 8])[0]
                        self.min_luminance = struct.unpack('<f', jxr_data[idx + 8:idx + 12])[0]
                        self.white_point = struct.unpack('<f', jxr_data[idx + 12:idx + 16])[0]
                    except struct.error:
                        # If metadata format is different, use defaults
                        pass
                    self.has_metadata = True
                    return True
        except Exception as e:
            logging.debug(f"No HDR metadata found: {e}")
        return False

    def to_dict(self) -> Dict:
        """Convert metadata to dictionary."""
        return {
            'max_luminance': self.max_luminance,
            'min_luminance': self.min_luminance,
            'white_point': self.white_point,
            'color_primaries': self.color_primaries,
            'transfer_function': self.transfer_function,
            'color_space': self.color_space,
            'bit_depth': self.bit_depth,
            'has_metadata': self.has_metadata
        }

import os
import subprocess
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# ============================================================
# CONFIGURATION AND CONSTANTS
# ============================================================
#
# Here we define configuration variables and constants used throughout the script.
# Adjust these values or paths as needed.
#

# Default parameters for HDRFix tone mapping
DEFAULT_TONE_MAP = "hable"
DEFAULT_PREGAMMA = "1.2"
DEFAULT_AUTOEXPOSURE = "0.9"

# Derive the current script directory to locate executables
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths to external tools, assumed to be in the same directory as this script
JXRDEC_PATH = os.path.join(CURRENT_DIR, "JXRDecApp.exe")
HDRFIX_PATH = os.path.join(CURRENT_DIR, "hdrfix.exe")

# Supported tone map methods (if you want to enforce user choices)
SUPPORTED_TONE_MAPS = {"hable", "reinhard", "filmic", "aces", "uncharted2"}

# Preview dimensions used for displaying before/after images in the GUI
PREVIEW_WIDTH = 512
PREVIEW_HEIGHT = 288


# ============================================================
# UTILITY FUNCTIONS
# ============================================================
#
# These functions handle the underlying logic such as converting JXR to TIFF,
# converting TIFF to PNG, running the HDRFix tool, and validating input/output.
# Keeping them separate from the GUI code makes it easier to maintain and
# potentially test them.
#

def decode_jxr_to_tiff(jxr_path: str) -> str:
    """
    Decode a JXR image file into a temporary TIFF file using JXRDecApp.exe.

    Parameters:
        jxr_path (str): The path to the input JXR file.

    Returns:
        str: The path to the generated temporary TIFF file.

    Raises:
        FileNotFoundError: If JXRDecApp.exe is missing or the input JXR file is not found.
        subprocess.CalledProcessError: If the decoding process fails.
    """
    if not os.path.exists(JXRDEC_PATH):
        raise FileNotFoundError("JXRDecApp.exe not found. Please ensure it is in the script directory.")

    if not os.path.exists(jxr_path):
        raise FileNotFoundError(f"Input JXR file not found at: {jxr_path}")

    # Create a temporary TIFF file
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tiff_path = tmp.name

    # Command to run JXRDecApp
    cmd = [JXRDEC_PATH, "-i", jxr_path, "-o", tiff_path]

    # Run the command
    result = subprocess.run(cmd, check=True, capture_output=True, text=True, shell=True)

    if not os.path.exists(tiff_path) or os.path.getsize(tiff_path) == 0:
        raise subprocess.CalledProcessError(
            returncode=result.returncode,
            cmd=cmd,
            stderr="Failed to create a valid TIFF file."
        )

    return tiff_path


def convert_tiff_to_png(tiff_path: str) -> str:
    """
    Convert a TIFF image to a PNG file. Attempts to use ImageMagick first, and if unavailable,
    falls back to Python's Pillow library.

    Parameters:
        tiff_path (str): The path to the input TIFF file.

    Returns:
        str: The path to the generated PNG file.

    Raises:
        RuntimeError: If the conversion fails using both ImageMagick and Pillow.
    """
    temp_png = tiff_path + ".png"

    # Try using ImageMagick (magick convert)
    try:
        subprocess.run(['magick', 'convert', tiff_path, temp_png],
                       check=True, capture_output=True, text=True)
        return temp_png
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Fallback to Pillow if ImageMagick is not available
    try:
        img = Image.open(tiff_path)
        img.save(temp_png, 'PNG')
        return temp_png
    except Exception as e:
        raise RuntimeError(f"Failed to convert TIFF to PNG: {str(e)}")


def run_hdrfix(input_file: str, output_file: str, tone_map: str, pre_gamma: str, auto_exposure: str) -> None:
    """
    Run hdrfix.exe to convert the decoded JXR file to an SDR JPEG using the specified parameters.

    Parameters:
        input_file (str): The path to the decoded image file (usually JXR or TIFF).
        output_file (str): The desired path for the output SDR JPEG file.
        tone_map (str): The tone mapping method.
        pre_gamma (str): The gamma value to apply before tone mapping.
        auto_exposure (str): The auto-exposure parameter.

    Raises:
        FileNotFoundError: If hdrfix.exe is not found in the script directory.
        subprocess.CalledProcessError: If hdrfix.exe fails to run or conversion fails.
    """
    if not os.path.exists(HDRFIX_PATH):
        raise FileNotFoundError("hdrfix.exe not found. Please ensure it is in the script directory.")

    cmd = [
        HDRFIX_PATH, input_file, output_file,
        "--tone-map", tone_map,
        "--pre-gamma", pre_gamma,
        "--auto-exposure", auto_exposure
    ]

    subprocess.run(cmd, check=True, capture_output=True, text=True)


def validate_files(input_file: str, output_file: str) -> None:
    """
    Validate that the input file exists and the output file location is writable.

    Parameters:
        input_file (str): The path to the input JXR file.
        output_file (str): The path to the output JPEG file.

    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If the output file path is empty.
    """
    if not input_file or not os.path.exists(input_file):
        raise FileNotFoundError("Input file does not exist or not specified.")

    if not output_file:
        raise ValueError("Output file path is not specified.")


def validate_parameters(tone_map: str, pre_gamma: str, auto_exposure: str) -> None:
    """
    Validate the parameters passed to hdrfix.

    Parameters:
        tone_map (str): The tone mapping method.
        pre_gamma (str): The gamma value before tone mapping.
        auto_exposure (str): The auto-exposure parameter.

    Raises:
        ValueError: If parameters are out of range, not numeric (where required),
                    or not supported.
    """
    if tone_map not in SUPPORTED_TONE_MAPS:
        raise ValueError(f"Tone map '{tone_map}' is not supported. Supported: {SUPPORTED_TONE_MAPS}")

    # Validate numeric parameters
    try:
        float(pre_gamma)
        float(auto_exposure)
    except ValueError:
        raise ValueError("pre-gamma and auto-exposure must be numeric values.")


# ============================================================
# GUI CLASS
# ============================================================
#
# The HDRFixGUI class handles the graphical user interface. It provides
# fields for selecting input/output files, adjusting parameters, displaying
# previews before and after conversion, and running the conversion process.
#

class HDRFixGUI:
    """
    A GUI application for converting JXR images to JPG using HDRFix and JXRDecApp.
    It allows browsing for input JXR, specifying output JPG, and setting tone mapping,
    pre-gamma, and auto-exposure parameters.
    """

    def __init__(self, master):
        """
        Initialize the GUI, creating all frames, labels, entries, and buttons.
        """
        self.master = master
        master.title("HDRFix Converter")
        master.resizable(False, False)

        # Main container frame
        main_frame = tk.Frame(master)
        main_frame.grid(row=0, column=0, sticky="nsew")

        # Left-side frame (controls)
        left_frame = tk.Frame(main_frame)
        left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nw")

        # Right-side frame (previews)
        right_frame = tk.Frame(main_frame)
        right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nw")

        self._setup_file_selection(left_frame)
        self._setup_parameters(left_frame)
        self._setup_conversion_button(left_frame)
        self._setup_status_label(left_frame)
        self._setup_previews(right_frame)

        # Variables for holding image references to prevent garbage collection
        self.before_image_ref = None
        self.after_image_ref = None

    def _setup_file_selection(self, parent_frame):
        """
        Setup the file selection entries and browse buttons.
        """
        file_frame = tk.LabelFrame(parent_frame, text="File Selection", padx=10, pady=10)
        file_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        tk.Label(file_frame, text="Input JXR:").grid(row=0, column=0, sticky="e")
        self.input_entry = tk.Entry(file_frame, width=40)
        self.input_entry.grid(row=0, column=1, padx=5)
        tk.Button(file_frame, text="Browse...", command=self.browse_input).grid(row=0, column=2, padx=5)

        tk.Label(file_frame, text="Output JPG:").grid(row=1, column=0, sticky="e")
        self.output_entry = tk.Entry(file_frame, width=40)
        self.output_entry.grid(row=1, column=1, padx=5)
        tk.Button(file_frame, text="Browse...", command=self.browse_output).grid(row=1, column=2, padx=5)

    def _setup_parameters(self, parent_frame):
        """
        Setup the parameter input fields (tone-map, pre-gamma, auto-exposure).
        """
        params_frame = tk.LabelFrame(parent_frame, text="Parameters", padx=10, pady=10)
        params_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))

        tk.Label(params_frame, text="Tone Map:").grid(row=0, column=0, sticky="e")
        self.tonemap_var = tk.StringVar(value=DEFAULT_TONE_MAP)
        tk.Entry(params_frame, textvariable=self.tonemap_var, width=15).grid(row=0, column=1, padx=5)

        tk.Label(params_frame, text="Pre-gamma:").grid(row=1, column=0, sticky="e")
        self.pregamma_var = tk.StringVar(value=DEFAULT_PREGAMMA)
        tk.Entry(params_frame, textvariable=self.pregamma_var, width=15).grid(row=1, column=1, padx=5)

        tk.Label(params_frame, text="Auto-exposure:").grid(row=2, column=0, sticky="e")
        self.autoexposure_var = tk.StringVar(value=DEFAULT_AUTOEXPOSURE)
        tk.Entry(params_frame, textvariable=self.autoexposure_var, width=15).grid(row=2, column=1, padx=5)

    def _setup_conversion_button(self, parent_frame):
        """
        Setup the conversion button that triggers the HDR->SDR conversion process.
        """
        convert_btn = tk.Button(parent_frame, text="Convert", command=self.convert_image, bg="#4CAF50", fg="white")
        convert_btn.grid(row=2, column=0, pady=10, sticky="ew")

    def _setup_status_label(self, parent_frame):
        """
        Setup a label to display status messages and errors to the user.
        """
        self.status_label = tk.Label(parent_frame, text="", fg="blue")
        self.status_label.grid(row=3, column=0, sticky="w")

    def _setup_previews(self, parent_frame):
        """
        Setup the frames and canvases for the before and after previews.
        """
        previews_frame = tk.Frame(parent_frame)
        previews_frame.grid(row=0, column=0, sticky="nsew")

        # Before Conversion Frame
        before_frame = tk.LabelFrame(previews_frame, text="Before Conversion")
        before_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        before_canvas = tk.Canvas(before_frame, width=PREVIEW_WIDTH, height=PREVIEW_HEIGHT)
        before_canvas.pack(padx=10, pady=10)
        self.before_label = tk.Label(before_canvas)
        before_canvas.create_window(PREVIEW_WIDTH//2, PREVIEW_HEIGHT//2, window=self.before_label)

        # After Conversion Frame
        after_frame = tk.LabelFrame(previews_frame, text="After Conversion")
        after_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        after_canvas = tk.Canvas(after_frame, width=PREVIEW_WIDTH, height=PREVIEW_HEIGHT)
        after_canvas.pack(padx=10, pady=10)
        self.after_label = tk.Label(after_canvas)
        after_canvas.create_window(PREVIEW_WIDTH//2, PREVIEW_HEIGHT//2, window=self.after_label)

    def browse_input(self):
        """
        Open a file dialog to select an input JXR file. Attempt to load a preview of it.
        """
        filename = filedialog.askopenfilename(
            title="Select Input JXR File",
            filetypes=[("JXR files", "*.jxr"), ("All files", "*.*")]
        )
        if filename:
            self.input_entry.delete(0, tk.END)
            self.input_entry.insert(0, filename)
            self.status_label.config(text="Loading preview...", fg="blue")
            self.master.update_idletasks()
            self.create_preview_from_jxr(filename)

    def browse_output(self):
        """
        Open a file dialog to select the output JPG file.
        """
        filename = filedialog.asksaveasfilename(
            title="Select Output JPG File",
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        if filename:
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, filename)

    def create_preview_from_jxr(self, jxr_file: str):
        """
        Create a 'before' preview from the selected JXR file by:
        1. Decoding it to TIFF.
        2. Displaying it in the "Before Conversion" frame.
        """
        if not jxr_file.lower().endswith('.jxr'):
            self.status_label.config(text="Not a JXR file", fg="red")
            self.before_label.config(image="", text="No Preview")
            self.before_image_ref = None
            return

        tiff_path = None
        try:
            tiff_path = decode_jxr_to_tiff(jxr_file)
            self.show_preview_from_file(tiff_path, is_before=True)
            self.status_label.config(text="Preview loaded successfully", fg="green")
        except FileNotFoundError as e:
            self.status_label.config(text=str(e), fg="red")
            messagebox.showerror("Error", str(e))
            self.before_label.config(image="", text="No Preview")
            self.before_image_ref = None
        except subprocess.CalledProcessError:
            self.status_label.config(text="Preview failed: JXRDecApp error", fg="red")
            self.before_label.config(image="", text="Preview Failed")
            self.before_image_ref = None
        except Exception as e:
            self.status_label.config(text=f"Preview failed: {str(e)}", fg="red")
            self.before_label.config(image="", text="Preview Failed")
            self.before_image_ref = None
        finally:
            # Cleanup temporary TIFF
            if tiff_path and os.path.exists(tiff_path):
                try:
                    os.remove(tiff_path)
                except:
                    pass

    def show_preview_from_file(self, filepath: str, is_before: bool):
        """
        Show a preview of the given image file (TIFF/JPG/PNG) in the appropriate label.

        If TIFF, converts it to PNG first. Resizes the image to fit the preview area.
        """
        temp_png = None
        label = self.before_label if is_before else self.after_label
        try:
            # If TIFF, convert to PNG for display
            if filepath.lower().endswith(('.tif', '.tiff')):
                temp_png = convert_tiff_to_png(filepath)
                filepath = temp_png

            img = Image.open(filepath).convert('RGB')

            # Resize the image to fit PREVIEW_WIDTH x PREVIEW_HEIGHT
            img_ratio = img.width / img.height
            target_ratio = PREVIEW_WIDTH / PREVIEW_HEIGHT

            if img_ratio > target_ratio:
                # Image is relatively wider than preview frame
                new_size = (PREVIEW_WIDTH, int(PREVIEW_WIDTH / img_ratio))
            else:
                # Image is relatively taller than preview frame
                new_size = (int(PREVIEW_HEIGHT * img_ratio), PREVIEW_HEIGHT)

            img = img.resize(new_size, Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)

            label.config(image=img_tk, text="")
            if is_before:
                self.before_image_ref = img_tk
            else:
                self.after_image_ref = img_tk

        except Exception as e:
            label_text = f"Preview Error: {str(e)}"
            label.config(image="", text=label_text)
            if is_before:
                self.before_image_ref = None
            else:
                self.after_image_ref = None
        finally:
            # Cleanup temporary PNG if created
            if temp_png and os.path.exists(temp_png):
                try:
                    os.remove(temp_png)
                except:
                    pass

    def convert_image(self):
        """
        Convert the selected JXR file to a JPG using HDRFix and the given parameters.
        On success, show the "After Conversion" preview.
        """
        input_file = self.input_entry.get().strip()
        output_file = self.output_entry.get().strip()
        tone_map = self.tonemap_var.get().strip()
        pre_gamma = self.pregamma_var.get().strip()
        auto_exposure = self.autoexposure_var.get().strip()

        # Validate files and parameters before running conversion
        try:
            validate_files(input_file, output_file)
            validate_parameters(tone_map, pre_gamma, auto_exposure)
        except (FileNotFoundError, ValueError) as e:
            messagebox.showerror("Error", str(e))
            self.status_label.config(text=str(e), fg="red")
            return

        self.status_label.config(text="Converting...", fg="blue")
        self.master.update_idletasks()

        try:
            # Run hdrfix conversion
            run_hdrfix(input_file, output_file, tone_map, pre_gamma, auto_exposure)
            self.status_label.config(text="Conversion successful!", fg="green")

            # Preview the output if it exists
            if os.path.exists(output_file):
                self.show_preview_from_file(output_file, is_before=False)
        except FileNotFoundError as e:
            error_msg = str(e)
            self.status_label.config(text=error_msg, fg="red")
            messagebox.showerror("Error", error_msg)
        except subprocess.CalledProcessError as e:
            error_msg = f"Conversion failed: {e.stderr if e.stderr else 'Unknown error'}"
            self.status_label.config(text=error_msg, fg="red")
            messagebox.showerror("Error", error_msg)
        except Exception as e:
            error_msg = f"Conversion failed: {str(e)}"
            self.status_label.config(text=error_msg, fg="red")
            messagebox.showerror("Error", error_msg)


# ============================================================
# MAIN ENTRY POINT
# ============================================================
#
# Run the application if this file is executed as a script.
#

if __name__ == "__main__":
    root = tk.Tk()
    app = HDRFixGUI(root)
    root.mainloop()

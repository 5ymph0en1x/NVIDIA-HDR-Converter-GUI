import os
import subprocess
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import TKinterModernThemes as TKMT
import concurrent.futures
import threading

# ============================================================
# CONFIGURATION AND CONSTANTS
# ============================================================
DEFAULT_TONE_MAP = "hable"
DEFAULT_PREGAMMA = "1.2"
DEFAULT_AUTOEXPOSURE = "0.9"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
JXRDEC_PATH = os.path.join(CURRENT_DIR, "JXRDecApp.exe")
HDRFIX_PATH = os.path.join(CURRENT_DIR, "hdrfix.exe")

SUPPORTED_TONE_MAPS = {"hable", "reinhard", "filmic", "aces", "uncharted2"}

PREVIEW_WIDTH = 512
PREVIEW_HEIGHT = 288


# ============================================================
# UTILITY FUNCTIONS (unchanged from previous code)
# ============================================================
def decode_jxr_to_tiff(jxr_path: str) -> str:
    if not os.path.exists(JXRDEC_PATH):
        raise FileNotFoundError("JXRDecApp.exe not found. Please ensure it is in the script directory.")

    if not os.path.exists(jxr_path):
        raise FileNotFoundError(f"Input JXR file not found at: {jxr_path}")

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tiff_path = tmp.name

    cmd = [JXRDEC_PATH, "-i", jxr_path, "-o", tiff_path]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True, shell=True)
    if not os.path.exists(tiff_path) or os.path.getsize(tiff_path) == 0:
        raise subprocess.CalledProcessError(
            returncode=result.returncode,
            cmd=cmd,
            stderr="Failed to create a valid TIFF file."
        )
    return tiff_path


def convert_tiff_to_png(tiff_path: str) -> str:
    temp_png = tiff_path + ".png"
    # Try using ImageMagick first
    try:
        subprocess.run(['magick', 'convert', tiff_path, temp_png],
                       check=True, capture_output=True, text=True)
        return temp_png
    except:
        pass

    # Fallback to Pillow
    try:
        img = Image.open(tiff_path)
        img.save(temp_png, 'PNG')
        return temp_png
    except Exception as e:
        raise RuntimeError(f"Failed to convert TIFF to PNG: {str(e)}")


def run_hdrfix(input_file: str, output_file: str, tone_map: str, pre_gamma: str, auto_exposure: str) -> None:
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
    if not input_file or not os.path.exists(input_file):
        raise FileNotFoundError("Input file does not exist or not specified.")
    if not output_file:
        raise ValueError("Output file path is not specified.")


def validate_parameters(tone_map: str, pre_gamma: str, auto_exposure: str) -> None:
    if tone_map not in SUPPORTED_TONE_MAPS:
        raise ValueError(f"Tone map '{tone_map}' is not supported. Supported: {SUPPORTED_TONE_MAPS}")
    try:
        float(pre_gamma)
        float(auto_exposure)
    except ValueError:
        raise ValueError("pre-gamma and auto-exposure must be numeric values.")


# ============================================================
# MAIN APP CLASS USING ThemedTKinterFrame
# ============================================================
class App(TKMT.ThemedTKinterFrame):
    def __init__(self, theme="park", mode="dark"):
        # Initialize the themed frame with title, theme, and mode
        super().__init__("NVIDIA HDR Converter", theme, mode)

        # Create a main container frame on self.master (which is a proper Tk root)
        main_frame = ttk.Frame(self.master, padding=(10, 10))
        main_frame.grid(row=0, column=0, sticky="nsew")

        # Configure grid weights for resizing
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

        # Create left and right frames within main_frame
        left_frame = ttk.Frame(main_frame, padding=(10, 10))
        left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nw")

        right_frame = ttk.Frame(main_frame, padding=(10, 10))
        right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nw")

        # Setup GUI components
        self._setup_mode_selection(left_frame)
        self._setup_file_selection(left_frame)
        self._setup_parameters(left_frame)
        self._setup_conversion_button(left_frame)
        self._setup_progress_bar(left_frame)
        self._setup_status_label(left_frame)
        self._setup_previews(right_frame)

        self.before_image_ref = None
        self.after_image_ref = None

        # Lock for thread-safe UI updates
        self.ui_lock = threading.Lock()

        # Finally, run the main loop with theming and resizing managed by TKMT
        self.run()

    def _setup_mode_selection(self, parent_frame):
        mode_frame = ttk.LabelFrame(parent_frame, text="Mode Selection", padding=(10, 10))
        mode_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        self.mode_var = tk.StringVar(value="single")

        single_radio = ttk.Radiobutton(mode_frame, text="Single File", variable=self.mode_var, value="single",
                                       command=self.update_mode)
        single_radio.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        folder_radio = ttk.Radiobutton(mode_frame, text="Folder", variable=self.mode_var, value="folder",
                                       command=self.update_mode)
        folder_radio.grid(row=0, column=1, padx=5, pady=5, sticky="w")

    def _setup_file_selection(self, parent_frame):
        self.file_frame = ttk.LabelFrame(parent_frame, text="File Selection", padding=(10, 10))
        self.file_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))

        # Input selection
        self.input_label = ttk.Label(self.file_frame, text="Input JXR:")
        self.input_label.grid(row=0, column=0, sticky="e", padx=(0, 5), pady=5)
        self.input_entry = ttk.Entry(self.file_frame, width=40)
        self.input_entry.grid(row=0, column=1, padx=(0, 5), pady=5)
        self.browse_button = ttk.Button(self.file_frame, text="Browse...", command=self.browse_input)
        self.browse_button.grid(row=0, column=2, padx=(0, 5), pady=5)

        # Output selection (only relevant for single file)
        self.output_label = ttk.Label(self.file_frame, text="Output JPG:")
        self.output_label.grid(row=1, column=0, sticky="e", padx=(0, 5), pady=5)
        self.output_entry = ttk.Entry(self.file_frame, width=40)
        self.output_entry.grid(row=1, column=1, padx=(0, 5), pady=5)
        self.output_browse_button = ttk.Button(self.file_frame, text="Browse...", command=self.browse_output)
        self.output_browse_button.grid(row=1, column=2, padx=(0, 5), pady=5)

        # Initially set to single file mode
        self.update_mode()

    def _setup_parameters(self, parent_frame):
        params_frame = ttk.LabelFrame(parent_frame, text="Parameters", padding=(10, 10))
        params_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))

        ttk.Label(params_frame, text="Tone Map:").grid(row=0, column=0, sticky="e", padx=(0, 5), pady=5)
        self.tonemap_var = tk.StringVar(value=DEFAULT_TONE_MAP)
        tone_map_values = sorted(list(SUPPORTED_TONE_MAPS))
        tone_map_dropdown = ttk.Combobox(params_frame, textvariable=self.tonemap_var, values=tone_map_values,
                                         state='readonly')
        tone_map_dropdown.grid(row=0, column=1, padx=(0, 5), pady=5)
        tone_map_dropdown.set(DEFAULT_TONE_MAP)

        ttk.Label(params_frame, text="Pre-gamma:").grid(row=1, column=0, sticky="e", padx=(0, 5), pady=5)
        self.pregamma_var = tk.StringVar(value=DEFAULT_PREGAMMA)
        ttk.Entry(params_frame, textvariable=self.pregamma_var, width=15).grid(row=1, column=1, padx=(0, 5), pady=5)

        ttk.Label(params_frame, text="Auto-exposure:").grid(row=2, column=0, sticky="e", padx=(0, 5), pady=5)
        self.autoexposure_var = tk.StringVar(value=DEFAULT_AUTOEXPOSURE)
        ttk.Entry(params_frame, textvariable=self.autoexposure_var, width=15).grid(row=2, column=1, padx=(0, 5), pady=5)

    def _setup_conversion_button(self, parent_frame):
        self.convert_btn = ttk.Button(parent_frame, text="Convert", command=self.convert_image)
        self.convert_btn.grid(row=3, column=0, pady=(10, 10), sticky="ew")

    def _setup_progress_bar(self, parent_frame):
        self.progress = ttk.Progressbar(parent_frame, orient="horizontal", mode="determinate")
        self.progress.grid(row=4, column=0, sticky="ew", pady=(0, 10))

    def _setup_status_label(self, parent_frame):
        self.status_label = ttk.Label(parent_frame, text="", foreground="#CCCCCC")
        self.status_label.grid(row=5, column=0, sticky="w", pady=(0, 10))

    def _setup_previews(self, parent_frame):
        previews_frame = ttk.Frame(parent_frame, padding=(10, 10))
        previews_frame.grid(row=0, column=0, sticky="nsew")

        before_frame = ttk.LabelFrame(previews_frame, text="Before Conversion", padding=(10, 10))
        before_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        before_canvas = tk.Canvas(before_frame, width=PREVIEW_WIDTH, height=PREVIEW_HEIGHT)
        before_canvas.pack(padx=10, pady=10)
        self.before_label = ttk.Label(before_canvas)
        before_canvas.create_window(PREVIEW_WIDTH // 2, PREVIEW_HEIGHT // 2, window=self.before_label)

        after_frame = ttk.LabelFrame(previews_frame, text="After Conversion", padding=(10, 10))
        after_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        after_canvas = tk.Canvas(after_frame, width=PREVIEW_WIDTH, height=PREVIEW_HEIGHT)
        after_canvas.pack(padx=10, pady=10)
        self.after_label = ttk.Label(after_canvas)
        after_canvas.create_window(PREVIEW_WIDTH // 2, PREVIEW_HEIGHT // 2, window=self.after_label)

    def update_mode(self):
        mode = self.mode_var.get()
        if mode == "single":
            self.file_frame.config(text="File Selection")
            self.input_label.config(text="Input JXR:")
            # Show output fields
            self.output_label.grid()
            self.output_entry.grid()
            self.output_browse_button.grid()
        else:
            self.file_frame.config(text="Folder Selection")
            self.input_label.config(text="Input Folder:")
            # Hide output fields
            self.output_label.grid_remove()
            self.output_entry.grid_remove()
            self.output_browse_button.grid_remove()
            # Clear output entry if any
            self.output_entry.delete(0, tk.END)
            # Clear previews since multiple files are involved
            self.before_label.config(image="", text="No Preview")
            self.before_image_ref = None
            self.after_label.config(image="", text="No Preview")
            self.after_image_ref = None

    def browse_input(self):
        mode = self.mode_var.get()
        if mode == "single":
            filename = filedialog.askopenfilename(
                title="Select Input JXR File",
                filetypes=[("JXR files", "*.jxr"), ("All files", "*.*")]
            )
            if filename:
                self.input_entry.delete(0, tk.END)
                self.input_entry.insert(0, filename)
                self.status_label.config(text="Loading preview...", foreground="#CCCCCC")
                self.master.update_idletasks()
                self.create_preview_from_jxr(filename)
        else:
            foldername = filedialog.askdirectory(title="Select Folder Containing JXR Files")
            if foldername:
                self.input_entry.delete(0, tk.END)
                self.input_entry.insert(0, foldername)
                self.status_label.config(text="Folder selected. Ready to convert all JXR files.", foreground="#CCCCCC")
                # Clear previews since multiple files are involved
                self.before_label.config(image="", text="No Preview")
                self.before_image_ref = None
                self.after_label.config(image="", text="No Preview")
                self.after_image_ref = None

    def browse_output(self):
        filename = filedialog.asksaveasfilename(
            title="Select Output JPG File",
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        if filename:
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, filename)

    def create_preview_from_jxr(self, jxr_file: str):
        if not jxr_file.lower().endswith('.jxr'):
            self.status_label.config(text="Not a JXR file", foreground="red")
            self.before_label.config(image="", text="No Preview")
            self.before_image_ref = None
            return

        tiff_path = None
        try:
            tiff_path = decode_jxr_to_tiff(jxr_file)
            self.show_preview_from_file(tiff_path, is_before=True)
            self.status_label.config(text="Preview loaded successfully", foreground="#00FF00")
        except FileNotFoundError as e:
            self.status_label.config(text=str(e), foreground="red")
            messagebox.showerror("Error", str(e))
            self.before_label.config(image="", text="No Preview")
            self.before_image_ref = None
        except subprocess.CalledProcessError:
            self.status_label.config(text="Preview failed: JXRDecApp error", foreground="red")
            self.before_label.config(image="", text="Preview Failed")
            self.before_image_ref = None
        except Exception as e:
            self.status_label.config(text=f"Preview failed: {str(e)}", foreground="red")
            self.before_label.config(image="", text="Preview Failed")
            self.before_image_ref = None
        finally:
            if tiff_path and os.path.exists(tiff_path):
                try:
                    os.remove(tiff_path)
                except:
                    pass

    def show_preview_from_file(self, filepath: str, is_before: bool):
        temp_png = None
        label = self.before_label if is_before else self.after_label
        try:
            if filepath.lower().endswith(('.tif', '.tiff')):
                temp_png = convert_tiff_to_png(filepath)
                filepath = temp_png

            img = Image.open(filepath).convert('RGB')
            img_ratio = img.width / img.height
            target_ratio = PREVIEW_WIDTH / PREVIEW_HEIGHT

            if img_ratio > target_ratio:
                new_size = (PREVIEW_WIDTH, int(PREVIEW_WIDTH / img_ratio))
            else:
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
            if temp_png and os.path.exists(temp_png):
                try:
                    os.remove(temp_png)
                except:
                    pass

    def convert_image(self):
        mode = self.mode_var.get()
        if mode == "single":
            self.convert_single_file()
        else:
            self.convert_folder()

    def convert_single_file(self):
        input_file = self.input_entry.get().strip()
        output_file = self.output_entry.get().strip()
        tone_map = self.tonemap_var.get().strip()
        pre_gamma = self.pregamma_var.get().strip()
        auto_exposure = self.autoexposure_var.get().strip()

        try:
            validate_files(input_file, output_file)
            validate_parameters(tone_map, pre_gamma, auto_exposure)
        except (FileNotFoundError, ValueError) as e:
            messagebox.showerror("Error", str(e))
            self.status_label.config(text=str(e), foreground="red")
            return

        self.status_label.config(text="Converting...", foreground="#CCCCCC")
        self.progress['value'] = 0
        self.master.update_idletasks()

        # Disable the Convert button
        self.convert_btn.config(state='disabled')

        def task():
            try:
                run_hdrfix(input_file, output_file, tone_map, pre_gamma, auto_exposure)
                self.safe_update_ui("Conversion successful!", "#00FF00")
                if os.path.exists(output_file):
                    self.show_preview_from_file(output_file, is_before=False)
            except FileNotFoundError as e:
                error_msg = str(e)
                self.safe_update_ui(error_msg, "red")
                messagebox.showerror("Error", error_msg)
            except subprocess.CalledProcessError as e:
                error_msg = f"Conversion failed: {e.stderr if e.stderr else 'Unknown error'}"
                self.safe_update_ui(error_msg, "red")
                messagebox.showerror("Error", error_msg)
            except Exception as e:
                error_msg = f"Conversion failed: {str(e)}"
                self.safe_update_ui(error_msg, "red")
                messagebox.showerror("Error", error_msg)
            finally:
                # Re-enable the Convert button
                self.safe_enable_convert_button()

        threading.Thread(target=task, daemon=True).start()

    def convert_folder(self):
        folder_path = self.input_entry.get().strip()
        tone_map = self.tonemap_var.get().strip()
        pre_gamma = self.pregamma_var.get().strip()
        auto_exposure = self.autoexposure_var.get().strip()

        if not folder_path or not os.path.isdir(folder_path):
            error_msg = "Please select a valid folder."
            messagebox.showerror("Error", error_msg)
            self.status_label.config(text=error_msg, foreground="red")
            return

        jxr_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.jxr')]
        if not jxr_files:
            error_msg = "No JXR files found in the selected folder."
            messagebox.showerror("Error", error_msg)
            self.status_label.config(text=error_msg, foreground="red")
            return

        output_folder = os.path.join(folder_path, "Converted_JPGs")
        os.makedirs(output_folder, exist_ok=True)

        self.status_label.config(text=f"Converting {len(jxr_files)} files...", foreground="#CCCCCC")
        self.progress['maximum'] = len(jxr_files)
        self.progress['value'] = 0
        self.master.update_idletasks()

        # Disable the Convert button
        self.convert_btn.config(state='disabled')

        def process_file(jxr_file):
            input_path = os.path.join(folder_path, jxr_file)
            output_filename = os.path.splitext(jxr_file)[0] + ".jpg"
            output_path = os.path.join(output_folder, output_filename)
            try:
                run_hdrfix(input_path, output_path, tone_map, pre_gamma, auto_exposure)
                self.safe_increment_progress()
            except Exception as e:
                self.safe_update_ui(f"Error processing {jxr_file}: {str(e)}", "red")

        def task():
            # Determine the number of CPU cores
            cpu_cores = os.cpu_count() or 1  # Defaults to 1 if os.cpu_count() returns None
            max_workers = max(1, cpu_cores // 2)  # Ensure at least 1 worker

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(process_file, f) for f in jxr_files]
                concurrent.futures.wait(futures)
            self.safe_update_ui("Batch conversion completed!", "#00FF00")
            # Re-enable the Convert button
            self.safe_enable_convert_button()

        threading.Thread(target=task, daemon=True).start()

    def safe_update_ui(self, message, color):
        with self.ui_lock:
            self.status_label.config(text=message, foreground=color)

    def safe_increment_progress(self):
        with self.ui_lock:
            self.progress['value'] += 1
            self.master.update_idletasks()

    def safe_enable_convert_button(self):
        with self.ui_lock:
            self.convert_btn.config(state='normal')


# ============================================================
# RUN THE APP
# ============================================================
if __name__ == "__main__":
    App("park", "dark")

import os
import subprocess
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import Image, ImageTk


def decode_jxr_to_tiff(jxr_path):
    """
    Decode the given JXR file into a TIFF file using JXRDecApp.exe.
    Returns the path to the generated temporary TIFF file.
    Raises FileNotFoundError or subprocess.CalledProcessError on failure.
    """
    # Create a temporary TIFF file
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tiff_path = tmp.name

    # Locate JXRDecApp.exe in the script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    jxrdec_path = os.path.join(current_dir, "JXRDecApp.exe")

    if not os.path.exists(jxrdec_path):
        raise FileNotFoundError("JXRDecApp.exe not found. Please ensure it is in the script directory.")

    if not os.path.exists(jxr_path):
        raise FileNotFoundError(f"Input JXR file not found at: {jxr_path}")

    # Run JXRDecApp to convert JXR to TIFF
    # Adjust parameters as needed; here we rely on defaults.
    cmd = [jxrdec_path, "-i", jxr_path, "-o", tiff_path]

    result = subprocess.run(cmd, check=True, capture_output=True, text=True, shell=True)

    if not os.path.exists(tiff_path) or os.path.getsize(tiff_path) == 0:
        raise subprocess.CalledProcessError(
            returncode=result.returncode, cmd=cmd, stderr="Failed to create a valid TIFF file."
        )

    return tiff_path


def convert_tiff_to_png(tiff_path):
    """
    Convert a TIFF image to PNG format for preview.
    Tries ImageMagick (magick convert) first, then Pillow if ImageMagick is unavailable.
    Returns the path to the temporary PNG file.
    """
    temp_png = tiff_path + ".png"
    # Try using ImageMagick first
    try:
        subprocess.run(['magick', 'convert', tiff_path, temp_png], check=True, capture_output=True, text=True)
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


class HDRFixGUI:
    """
    A GUI application for converting JXR images to JPG using HDRFix and JXRDecApp.
    It allows browsing for input JXR, specifying output JPG, and setting tone mapping,
    pre-gamma, and auto-exposure parameters.
    """

    def __init__(self, master):
        self.master = master
        master.title("HDRFix Converter")
        master.resizable(False, False)

        # Main container
        main_frame = tk.Frame(master)
        main_frame.grid(row=0, column=0, sticky="nsew")

        # Left-side frame (controls)
        left_frame = tk.Frame(main_frame)
        left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nw")

        # Right-side frame (previews)
        right_frame = tk.Frame(main_frame)
        right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nw")

        # File Selection Frame
        file_frame = tk.LabelFrame(left_frame, text="File Selection", padx=10, pady=10)
        file_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        tk.Label(file_frame, text="Input JXR:").grid(row=0, column=0, sticky="e")
        self.input_entry = tk.Entry(file_frame, width=40)
        self.input_entry.grid(row=0, column=1, padx=5)
        tk.Button(file_frame, text="Browse...", command=self.browse_input).grid(row=0, column=2, padx=5)

        tk.Label(file_frame, text="Output JPG:").grid(row=1, column=0, sticky="e")
        self.output_entry = tk.Entry(file_frame, width=40)
        self.output_entry.grid(row=1, column=1, padx=5)
        tk.Button(file_frame, text="Browse...", command=self.browse_output).grid(row=1, column=2, padx=5)

        # Parameters Frame
        params_frame = tk.LabelFrame(left_frame, text="Parameters", padx=10, pady=10)
        params_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))

        tk.Label(params_frame, text="Tone Map:").grid(row=0, column=0, sticky="e")
        self.tonemap_var = tk.StringVar(value="hable")
        tk.Entry(params_frame, textvariable=self.tonemap_var, width=15).grid(row=0, column=1, padx=5)

        tk.Label(params_frame, text="Pre-gamma:").grid(row=1, column=0, sticky="e")
        self.pregamma_var = tk.StringVar(value="1.2")
        tk.Entry(params_frame, textvariable=self.pregamma_var, width=15).grid(row=1, column=1, padx=5)

        tk.Label(params_frame, text="Auto-exposure:").grid(row=2, column=0, sticky="e")
        self.autoexposure_var = tk.StringVar(value="0.9")
        tk.Entry(params_frame, textvariable=self.autoexposure_var, width=15).grid(row=2, column=1, padx=5)

        # Convert button
        convert_btn = tk.Button(left_frame, text="Convert", command=self.convert_image, bg="#4CAF50", fg="white")
        convert_btn.grid(row=2, column=0, pady=10, sticky="ew")

        # Status label
        self.status_label = tk.Label(left_frame, text="", fg="blue")
        self.status_label.grid(row=3, column=0, sticky="w")

        # Previews side-by-side
        previews_frame = tk.Frame(right_frame)
        previews_frame.grid(row=0, column=0, sticky="nsew")

        # Before Conversion Frame
        before_frame = tk.LabelFrame(previews_frame, text="Before Conversion")
        before_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        # Reduced canvas size to 512x288
        before_canvas = tk.Canvas(before_frame, width=512, height=288)
        before_canvas.pack(padx=10, pady=10)
        self.before_label = tk.Label(before_canvas)
        before_canvas.create_window(256, 144, window=self.before_label)

        # After Conversion Frame (now to the right of the before_frame)
        after_frame = tk.LabelFrame(previews_frame, text="After Conversion")
        after_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        # Reduced canvas size to 512x288
        after_canvas = tk.Canvas(after_frame, width=512, height=288)
        after_canvas.pack(padx=10, pady=10)
        self.after_label = tk.Label(after_canvas)
        after_canvas.create_window(256, 144, window=self.after_label)

        self.before_image_ref = None
        self.after_image_ref = None

    def browse_input(self):
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
        filename = filedialog.asksaveasfilename(
            title="Select Output JPG File",
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        if filename:
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, filename)

    def create_preview_from_jxr(self, jxr_file):
        """
        Create a preview from the input JXR file by:
        1. Decoding it to a temporary TIFF.
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

    def show_preview_from_file(self, filepath, is_before=False):
        """
        Show a preview of the given image file (TIFF/JPG/PNG) in the appropriate label.
        Converts TIFF to PNG if needed.
        """
        temp_png = None
        try:
            # If TIFF, convert to PNG
            if filepath.lower().endswith(('.tif', '.tiff')):
                temp_png = convert_tiff_to_png(filepath)
                filepath = temp_png

            img = Image.open(filepath).convert('RGB')

            # New preview size
            preview_size = (512, 288)
            img_ratio = img.width / img.height
            target_ratio = preview_size[0] / preview_size[1]

            if img_ratio > target_ratio:
                # Image is relatively wider
                new_size = (preview_size[0], int(preview_size[0] / img_ratio))
            else:
                # Image is relatively taller
                new_size = (int(preview_size[1] * img_ratio), preview_size[1])

            img = img.resize(new_size, Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)

            label = self.before_label if is_before else self.after_label
            label.config(image=img_tk, text="")
            if is_before:
                self.before_image_ref = img_tk
            else:
                self.after_image_ref = img_tk

        except Exception as e:
            label_text = f"Preview Error: {str(e)}"
            if is_before:
                self.before_label.config(image="", text=label_text)
                self.before_image_ref = None
            else:
                self.after_label.config(image="", text=label_text)
                self.after_image_ref = None
        finally:
            # Cleanup temporary PNG
            if temp_png and os.path.exists(temp_png):
                try:
                    os.remove(temp_png)
                except:
                    pass

    def convert_image(self):
        """
        Convert the selected JXR file to JPG using hdrfix.exe and the given parameters.
        On success, show the output preview.
        """
        input_file = self.input_entry.get().strip()
        output_file = self.output_entry.get().strip()
        tone_map = self.tonemap_var.get().strip()
        pre_gamma = self.pregamma_var.get().strip()
        auto_exposure = self.autoexposure_var.get().strip()

        if not input_file or not output_file:
            messagebox.showerror("Error", "Please select both input and output files.")
            return

        if not os.path.exists(input_file):
            messagebox.showerror("Error", "Input file does not exist.")
            return

        # Get the directory of the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        hdrfix_path = os.path.join(current_dir, "hdrfix.exe")

        if not os.path.exists(hdrfix_path):
            self.status_label.config(text="hdrfix.exe not found", fg="red")
            messagebox.showerror(
                "Error",
                "Could not find hdrfix.exe. Please ensure it is in the same directory as this script."
            )
            return

        cmd = [
            hdrfix_path, input_file, output_file,
            "--tone-map", tone_map,
            "--pre-gamma", pre_gamma,
            "--auto-exposure", auto_exposure
        ]

        self.status_label.config(text="Converting...", fg="blue")
        self.master.update_idletasks()

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.status_label.config(text="Conversion successful!", fg="green")

            # Preview the output if it exists
            if os.path.exists(output_file):
                self.show_preview_from_file(output_file, is_before=False)
        except subprocess.CalledProcessError as e:
            error_msg = f"Conversion failed: {e.stderr if e.stderr else 'Unknown error'}"
            self.status_label.config(text=error_msg, fg="red")
            messagebox.showerror("Error", error_msg)



if __name__ == "__main__":
    root = tk.Tk()
    app = HDRFixGUI(root)
    root.mainloop()

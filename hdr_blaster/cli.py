"""The commands without the window: one capture or one recording converted, and the self-test of the built executable."""
import os

from . import configure_runtime
from .engine.constants import DEFAULT_RENDER_METHOD, SUPPORTED_TONE_MAPS
from .engine.device import DeviceManager
from .engine.pipeline import RenderSettings, export_image, render
from .engine.processor import HDRColorProcessor
from .engine.tonemap import AdvancedToneMapper
from .io.jxr import OptimizedJXRLoader


def _cli_convert(argv) -> int:
    """Headless one-shot conversion (bypasses the GUI).

        NHC.py --convert INPUT.jxr [--out BASE] [--method punchy]

    Without --out the result is written next to the source as INPUT_SDR.jpg, numbered
    when that exists: nothing is ever overwritten. Without --method the rendering is the
    application's default, Faithful; ``--method auto`` lets the engine pick an operator.
    Used as the bundled-exe self-test and as a minimal CLI. Returns an exit code.
    """
    import argparse
    parser = argparse.ArgumentParser(prog="NHC.py", add_help=True)
    parser.add_argument("--selftest", "--convert", dest="input", metavar="INPUT.jxr", required=True)
    parser.add_argument("--out", dest="out", default=None, help="output base path (no extension)")
    parser.add_argument("--method", dest="method", default=DEFAULT_RENDER_METHOD,
                        choices=sorted(SUPPORTED_TONE_MAPS | {"auto"}),
                        help="look or operator (default: faithful; 'auto' lets the engine pick an operator)")
    parser.add_argument("--white", type=float, default=None,
                        help="what SDR white stands for, in nits, for --method faithful (default: this screen's)")
    parser.add_argument("--color", type=float, default=0.0, help="AI Color (3D LUT) strength 0-100")
    parser.add_argument("--edge", type=float, default=25.0, help="Sharpness 0-100")
    parser.add_argument("--format", dest="fmt", default="JPEG", choices=["JPEG", "TIFF", "Both"])
    args = parser.parse_args(argv)
    configure_runtime()  # no-op when the launcher already did it

    dev = DeviceManager().get_device()
    loader = OptimizedJXRLoader(dev)
    lin, err, _ = loader.load_jxr(args.input)
    if lin is None:
        print(f"[selftest] FAILED to load {args.input}: {err}")
        return 2
    tm = AdvancedToneMapper(dev, loader.metadata)
    proc = HDRColorProcessor(dev, loader, use_fp16=False)
    chosen = None if args.method == "auto" else args.method      # None: the Auto-select oracle
    settings = RenderSettings(method=chosen,
                              color_strength=args.color, edge_strength=args.edge,
                              white_nits=args.white or _screen_white())
    enhanced, _ = render(tm, loader, proc, lin, settings)
    method = chosen or "auto-detect"
    if args.out:
        out_base = args.out
    else:
        from .session import Options, output_base

        out_base = output_base(args.input, "image", Options())    # INPUT_SDR, never over a file
    export_image(proc, lin, enhanced, out_base, args.fmt, include_raw=False)
    print(f"[selftest] OK  device={dev.type}  method={method}  -> {out_base}.jpg")
    return 0


def _screen_white() -> float:
    from .display import sdr_white_nits

    return float(sdr_white_nits()[0])


def _cli_convert_video(argv) -> int:
    """Headless video conversion.

        NHC.py --convert-video INPUT.mp4 [--out OUTPUT.mp4] [--method punchy]
               [--quality high|balanced|compact] [--height 1080] [--color 0] [--edge 25]

    Without --out the result is written next to the source as INPUT_SDR.mp4 (never
    overwriting). Returns an exit code: 0 done, 2 unsupported input, 1 other failure.
    """
    import argparse
    parser = argparse.ArgumentParser(prog="NHC.py", add_help=True)
    parser.add_argument("--convert-video", dest="input", metavar="INPUT", required=True)
    parser.add_argument("--out", dest="out", default=None, help="output file (default: INPUT_SDR.mp4)")
    parser.add_argument("--method", dest="method", default=DEFAULT_RENDER_METHOD,
                        choices=sorted(SUPPORTED_TONE_MAPS | {"auto"}),
                        help="look or operator (default: faithful; 'auto' picks one operator for the whole video)")
    parser.add_argument("--white", type=float, default=None,
                        help="what SDR white stands for, in nits, for --method faithful (default: this screen's)")
    parser.add_argument("--quality", default="high", choices=["high", "balanced", "compact"])
    parser.add_argument("--height", type=int, default=None, help="scale down to this height, e.g. 1080")
    parser.add_argument("--color", type=float, default=0.0, help="AI Color (3D LUT) strength 0-100")
    parser.add_argument("--edge", type=float, default=25.0, help="Sharpness 0-100")
    args = parser.parse_args(argv)
    configure_runtime()

    from .video.job import VideoSettings, convert_video, default_output_path
    from .video.reader import UnsupportedVideo

    dev = DeviceManager().get_device()
    out = args.out or default_output_path(args.input)
    settings = VideoSettings(
        render=RenderSettings(method=None if args.method == "auto" else args.method,
                              color_strength=args.color, edge_strength=args.edge,
                              white_nits=args.white or _screen_white()),
        quality=args.quality, out_height=args.height)
    last = {"stage": None, "tenth": -1}

    def progress(stage, done, total):
        tenth = done * 10 // total if total else done // 300
        if stage != last["stage"] or tenth != last["tenth"]:
            last.update(stage=stage, tenth=tenth)
            print(f"[video] {stage}: {done}/{total or '?'}", flush=True)

    try:
        report = convert_video(args.input, out, settings, dev, progress=progress)
    except UnsupportedVideo as exc:
        print(f"[video] {exc}")
        return 2
    except Exception as exc:
        print(f"[video] FAILED: {exc}")
        return 1
    megabytes = lambda n: f"{n / 1e6:.1f} MB"
    print(f"[video] OK  {report.frames} frames in {report.seconds:.1f} s ({report.fps:.1f} fps)  "
          f"method={report.method}  encoder={report.encoder}  shots={len(report.cuts)}  "
          f"{megabytes(report.src_bytes)} -> {megabytes(report.dst_bytes)}  -> {report.dst}")
    return 0

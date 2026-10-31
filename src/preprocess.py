"""Stage 1 — Sketch normalization (v2).
Handles ALL input shapes (Gradio Sketchpad dict / PIL / ndarray), rejects blank canvases,
denoises, binarizes with a MANDATORY faint-sketch fallback + morphological cleanup,
and produces a white-strokes-on-black 512x512 map for ControlNet-Scribble."""
import cv2
import numpy as np
from PIL import Image
from config import IMAGE_SIZE, MIN_STROKE_COVERAGE

BLANK_STD_THRESHOLD = 2.0  # grayscale std below this = blank canvas


def normalize_sketch_input(x):
    """Accepts Gradio Sketchpad dict {'composite':...}, PIL.Image, or ndarray -> RGB PIL.Image."""
    if isinstance(x, dict):  # gr.Sketchpad / ImageEditor payload
        arr = x.get("composite")
        if arr is None and x.get("background") is not None:
            arr = x["background"]
        if arr is None:
            raise ValueError("Empty sketch input (canvas is blank).")
        x = arr
    if isinstance(x, np.ndarray):
        if x.ndim == 3 and x.shape[2] == 4:  # RGBA ndarray -> composite on white
            alpha = x[..., 3:4].astype(np.float32) / 255.0
            rgb = x[..., :3].astype(np.float32)
            x = (rgb * alpha + 255.0 * (1 - alpha)).astype(np.uint8)
        return Image.fromarray(x.astype(np.uint8), "RGB")
    if isinstance(x, Image.Image):
        if x.mode == "RGBA":
            bg = Image.new("RGB", x.size, (255, 255, 255))
            bg.paste(x, mask=x.split()[3])
            return bg
        return x.convert("RGB")
    raise TypeError(f"Unsupported sketch input type: {type(x)}")


def _pad_square(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = max(h, w)
    canvas = np.full((s, s), 0, dtype=np.uint8)  # black padding matches black bg
    y0, x0 = (s - h) // 2, (s - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = img
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)


def preprocess_sketch(input_image, invert_if_light_bg: bool = True):
    """RGBA composite -> blank check -> grayscale -> bilateral denoise -> Otsu (with
    MANDATORY faint-sketch fallback) -> morphological cleanup -> auto-invert to white
    strokes on black -> 512x512 RGB PIL."""
    pil = normalize_sketch_input(input_image)
    gray = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2GRAY)

    # Blank-canvas rejection: cheap guard BEFORE any GPU work
    if float(gray.std()) < BLANK_STD_THRESHOLD:
        raise ValueError("Canvas appears blank — please draw or upload a sketch.")

    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)  # kills paper texture, keeps strokes

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    stroke_coverage = float((binary == 0).mean())  # dark-pixel fraction

    if stroke_coverage < MIN_STROKE_COVERAGE:
        # Faint sketch: Otsu destroyed it. Use normalized inverted grayscale instead.
        norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        scribble = 255 - norm
    else:
        scribble = binary
        # Morphological cleanup: opening removes specks, dilation closes stroke gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        scribble = cv2.morphologyEx(scribble, cv2.MORPH_OPEN, kernel)
        scribble = cv2.morphologyEx(scribble, cv2.MORPH_DILATE, kernel)

    if invert_if_light_bg and scribble.mean() > 127:
        scribble = 255 - scribble  # ensure WHITE strokes on BLACK background

    scribble = _pad_square(scribble, IMAGE_SIZE)
    return Image.fromarray(cv2.cvtColor(scribble, cv2.COLOR_GRAY2RGB))
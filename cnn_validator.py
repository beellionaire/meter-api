import argparse
import json
import os
import re
import sys
from pathlib import Path

# Matikan pesan error TensorFlow yang berisik
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
import tensorflow as tf

def parse_args():
    parser = argparse.ArgumentParser(description="CNN validator untuk digit meter air.")
    parser.add_argument("image_path", help="Gambar meter.")
    parser.add_argument("model_path", help="Model CNN .keras/.h5.")
    parser.add_argument("previous_meter", type=int, help="Meter sebelumnya.")
    parser.add_argument("processed_dir", help="Folder output preview.")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--max-digits", type=int, default=6)
    return parser.parse_args()

def ensure_dir(path_str):
    path = Path(path_str)
    path.mkdir(parents=True, exist_ok=True)
    return path

def normalize_digits(text):
    digits = re.sub(r"\D", "", text or "")
    if digits == "": return ""
    return digits.lstrip("0") or "0"

def load_image(path_str):
    image = cv2.imread(path_str)
    if image is None: raise RuntimeError("Gambar tidak dapat dibaca oleh OpenCV.")
    return image

def preprocess_roi(image):
    # Ubah ke Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Binary untuk deteksi kontur (segmentasi)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return gray, binary

def segment_digits(binary):
    # Bersihkan noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height, width = binary.shape[:2]
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Filter kotak yang masuk akal (Tinggi minimal 20% tinggi ROI)
        if h > height * 0.20 and w > width * 0.01:
            boxes.append((x, y, w, h))

    # Urutkan dari kiri ke kanan
    boxes.sort(key=lambda box: box[0])
    
    # Gabungkan kotak yang berdekatan
    merged = []
    for box in boxes:
        if not merged:
            merged.append(box)
        else:
            px, py, pw, ph = merged[-1]
            x, y, w, h = box
            if x < (px + pw + 5):
                nx = min(px, x); ny = min(py, y)
                nr = max(px + pw, x + w); nb = max(py + ph, y + h)
                merged[-1] = (nx, ny, nr - nx, nb - ny)
            else:
                merged.append(box)
    return merged

def crop_digit(gray_image, box, image_size):
    x, y, w, h = box
    pad = 2
    x0, y0 = max(x-pad, 0), max(y-pad, 0)
    x1, y1 = min(x+w+pad, gray_image.shape[1]), min(y+h+pad, gray_image.shape[0])
    
    # 🔥 PENTING: Potong dari GRAY_IMAGE, bukan binary!
    digit = gray_image[y0:y1, x0:x1]
    
    # Buat latar putih (255) agar AI tidak bingung
    canvas_size = max(digit.shape[:2]) + 4
    canvas = np.full((canvas_size, canvas_size), 255, dtype=np.uint8)
    
    y_off = (canvas_size - digit.shape[0]) // 2
    x_off = (canvas_size - digit.shape[1]) // 2
    canvas[y_off:y_off+digit.shape[0], x_off:x_off+digit.shape[1]] = digit
    
    # Resize ke 64x64
    resized = cv2.resize(canvas, (image_size, image_size), interpolation=cv2.INTER_AREA)
    
    # 🔥 INI KUNCINYA: Jika modelmu dilatih angka Hitam di latar Putih, 
    # maka biarkan seperti ini. Jika modelmu menebak ngaco, buka komentar bawah ini:
    # resized = cv2.bitwise_not(resized)
    
    return resized

def predict_digits(model, gray_image, boxes, image_size):
    digit_images = [crop_digit(gray_image, box, image_size) for box in boxes]
    batch = np.asarray(digit_images, dtype=np.float32) / 255.0 # Normalisasi
    batch = np.expand_dims(batch, axis=-1)
    
    predictions = model.predict(batch, verbose=0)
    digits = [str(int(np.argmax(p))) for p in predictions]
    confs = [float(np.max(p) * 100) for p in predictions]
    return "".join(digits), confs, digit_images

def main():
    args = parse_args()
    source = load_image(args.image_path)
    gray, binary = preprocess_roi(source)
    boxes = segment_digits(binary)
    boxes = boxes[:args.max_digits]

    model = tf.keras.models.load_model(args.model_path)
    
    digits = ""
    conf = 0
    if boxes:
        raw_digits, confs, _ = predict_digits(model, gray, boxes, args.image_size)
        digits = normalize_digits(raw_digits)
        conf = round(float(np.mean(confs)), 2)

    print(json.dumps({
        "success": True,
        "normalized_text": digits,
        "confidence": conf,
        "is_valid": len(digits) >= 4,
        "engine": "cnn-validator-final"
    }))

if __name__ == "__main__":
    try: main()
    except Exception as e:
        print(json.dumps({"success": False, "message": str(e)}))
        sys.exit(1)
import argparse
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
import tensorflow as tf

def parse_args():
    parser = argparse.ArgumentParser(description="CNN validator untuk digit meter air.")
    parser.add_argument("image_path", help="Gambar meter, ROI, atau hasil preprocessing.")
    parser.add_argument("model_path", help="Model CNN .keras/.h5.")
    parser.add_argument("previous_meter", type=int, help="Meter sebelumnya.")
    parser.add_argument("processed_dir", help="Folder output preview segmentasi.")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--max-digits", type=int, default=6)
    return parser.parse_args()

def ensure_dir(path_str):
    path = Path(path_str)
    path.mkdir(parents=True, exist_ok=True)
    return path

def normalize_digits(text):
    digits = re.sub(r"\D", "", text or "")
    if digits == "":
        return ""
    trimmed = digits.lstrip("0")
    return "0" if trimmed == "" else trimmed

def load_image(path_str):
    image = cv2.imread(path_str)
    if image is None:
        raise RuntimeError("Gambar tidak dapat dibaca oleh OpenCV.")
    return image

def preprocess_roi(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # Binary Inverse HANYA digunakan untuk mencari kontur/kotak, BUKAN untuk dibaca AI
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return gray, binary

def segment_digits(binary):
    # Gunakan kernel yang lebih kecil agar tidak menghapus detail angka
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height, width = binary.shape[:2]
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # FILTER YANG LEBIH LONGGAR (Supaya tidak ada digit yang dibuang)
        # Kita hanya membuang yang ukurannya sangat kecil (noise/bintik)
        if h > height * 0.15 and w > width * 0.02: 
            boxes.append((x, y, w, h))

    if not boxes:
        return []

    # Sort dari kiri ke kanan
    boxes.sort(key=lambda box: box[0])
    
    # Gabungkan kotak yang berdekatan (karena angka mungkin terpotong garis)
    merged = []
    for box in boxes:
        if not merged:
            merged.append(box)
            continue
        px, py, pw, ph = merged[-1]
        x, y, w, h = box
        # Jika kotak tumpang tindih atau sangat dekat, gabungkan
        if x < (px + pw + 5):
            nx = min(px, x)
            ny = min(py, y)
            nr = max(px + pw, x + w)
            nb = max(py + ph, y + h)
            merged[-1] = (nx, ny, nr - nx, nb - ny)
        else:
            merged.append(box)
            
    return merged

def crop_digit(gray_image, box, image_size):
    x, y, w, h = box
    pad = max(int(max(w, h) * 0.18), 3)
    x0 = max(x - pad, 0)
    y0 = max(y - pad, 0)
    x1 = min(x + w + pad, gray_image.shape[1])
    y1 = min(y + h + pad, gray_image.shape[0])
    
    # Memotong dari gambar ABU-ABU (Grayscale), BUKAN gambar hitam putih invert
    digit = gray_image[y0:y1, x0:x1]

    canvas_size = max(digit.shape[:2]) + 8
    
    # 🔥 KUNCI PERBAIKAN 11111: KANVAS PUTIH (255) BUKAN HITAM (0) 🔥
    # Agar serasi dengan latar belakang kertas meteran asli
    canvas = np.full((canvas_size, canvas_size), 255, dtype=np.uint8) 
    
    y_offset = (canvas_size - digit.shape[0]) // 2
    x_offset = (canvas_size - digit.shape[1]) // 2
    canvas[y_offset:y_offset + digit.shape[0], x_offset:x_offset + digit.shape[1]] = digit
    resized = cv2.resize(canvas, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return resized

def predict_digits(model, gray_image, boxes, image_size):
    # Kirim gray_image ke crop_digit, BUKAN binary!
    digit_images = [crop_digit(gray_image, box, image_size) for box in boxes]
    batch = np.asarray(digit_images, dtype=np.float32)
    batch = np.expand_dims(batch, axis=-1)
    
    predictions = model.predict(batch, verbose=0)

    digits = []
    confidences = []
    for prediction in predictions:
        label = int(np.argmax(prediction))
        confidence = float(np.max(prediction) * 100.0)
        digits.append(str(label))
        confidences.append(confidence)

    return "".join(digits), confidences, digit_images

def save_preview(processed_dir, roi, binary, digit_images):
    out_dir = ensure_dir(processed_dir)
    token = os.urandom(6).hex()
    roi_path = out_dir / f"cnn_roi_{token}.png"
    binary_path = out_dir / f"cnn_binary_{token}.png"
    cv2.imwrite(str(roi_path), roi)
    cv2.imwrite(str(binary_path), binary)

    if digit_images:
        strip = np.concatenate(digit_images, axis=1)
        digits_path = out_dir / f"cnn_digits_{token}.png"
        cv2.imwrite(str(digits_path), strip)
    else:
        digits_path = None

    return roi_path, binary_path, digits_path

def validate(digits, previous_meter):
    errors = []
    if digits == "":
        errors.append("CNN tidak menemukan digit pada area ROI.")
    # Validasi panjang angka kita matikan batas ketatnya, kembalikan max-digits saja
    return errors

def relative_upload_path(path):
    if path is None:
        return None
    return "uploads/processed/" + Path(path).name

def main():
    args = parse_args()
    model_path = Path(args.model_path)
    if not model_path.is_file():
        raise RuntimeError(f"Model CNN tidak ditemukan: {model_path}")

    source = load_image(args.image_path)
    
    # 🔥 KUNCI PERBAIKAN "DITOLAK SERVER": MATIKAN DOUBLE CROPPING! 🔥
    # Karena PHP sudah memotong dengan benar, kita langsung pakai gambar aslinya sebagai ROI
    roi = source 
    
    gray, binary = preprocess_roi(roi)
    boxes = segment_digits(binary)
    boxes = boxes[:args.max_digits]

    model = tf.keras.models.load_model(model_path)
    if boxes:
        # Kirim GRAYSCALE (Abu-abu), BUKAN Binary Invert ke fungsi prediksi!
        raw_digits, confidences, digit_images = predict_digits(model, gray, boxes, args.image_size)
        digits = normalize_digits(raw_digits)
        confidence = round(float(np.mean(confidences)), 2) if confidences else None
    else:
        raw_digits = ""
        digits = ""
        confidence = None
        digit_images = []

    roi_path, binary_path, digits_path = save_preview(args.processed_dir, roi, binary, digit_images)
    errors = validate(digits, args.previous_meter)

    print(json.dumps({
        "success": True,
        "raw_text": raw_digits,
        "normalized_text": digits,
        "confidence": confidence,
        "is_valid": len(errors) == 0,
        "errors": errors,
        "engine": "cnn-validator",
        "roi_relative": relative_upload_path(roi_path),
        "processed_relative": relative_upload_path(digits_path or binary_path),
        "binary_relative": relative_upload_path(binary_path),
        "digit_count": len(boxes),
    }))

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({
            "success": False,
            "message": str(exc),
            "engine": "cnn-validator",
        }))
        sys.exit(1)
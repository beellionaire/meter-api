import os
import json
import re
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf

# Matikan log TensorFlow yang mengganggu
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

app = Flask(__name__)
CORS(app) # Mengizinkan Hostinger memanggil API ini tanpa terkena blokir CORS

# --- LOAD MODEL CNN SECARA GLOBAL (HANYA 1 KALI SAAT START) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "cnn_digit_meter.keras")

if os.path.exists(MODEL_PATH):
    print("🤖 Memuat Otak AI CNN...")
    cnn_model = tf.keras.models.load_model(MODEL_PATH)
else:
    print(f"⚠️ Model tidak ditemukan di: {MODEL_PATH}")
    cnn_model = None

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def prewitt_edges(gray):
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
    grad_x = cv2.filter2D(gray, cv2.CV_32F, kernel_x)
    grad_y = cv2.filter2D(gray, cv2.CV_32F, kernel_y)
    magnitude = cv2.magnitude(grad_x, grad_y)
    magnitude = cv2.convertScaleAbs(magnitude)
    _, edges = cv2.threshold(magnitude, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return edges

def find_digit_roi(image, gray, edges):
    ANCHOR_X, ANCHOR_Y, ANCHOR_W, ANCHOR_H = 0.28, 0.26, 0.45, 0.16
    h, w = gray.shape[:2]
    anchor_x = max(int(w * ANCHOR_X), 0)
    anchor_y = max(int(h * ANCHOR_Y), 0)
    anchor_w = min(int(w * ANCHOR_W), w - anchor_x)
    anchor_h = min(int(h * ANCHOR_H), h - anchor_y)

    search_gray = gray[anchor_y:anchor_y + anchor_h, anchor_x:anchor_x + anchor_w]
    search_edges = edges[anchor_y:anchor_y + anchor_h, anchor_x:anchor_x + anchor_w]
    sh, sw = search_gray.shape[:2]

    _, dark_mask = cv2.threshold(search_gray, 135, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    dark_mask = cv2.dilate(dark_mask, np.ones((3, 3), np.uint8), iterations=1)
    combined = cv2.bitwise_or(dark_mask, search_edges)
    contours, _ = cv2.findContours(combined, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    best_rect = None
    best_score = -1e18

    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        if cw < sw * 0.35 or ch < sh * 0.35: continue
        ratio = cw / float(max(ch, 1))
        if ratio < 2.8 or ratio > 6.8: continue
        
        area_ratio = (cw * ch) / float(max(sw * sh, 1))
        if area_ratio < 0.18 or area_ratio > 0.90: continue

        inner_brightness = float(np.mean(search_gray[y:y+ch, x:x+cw])) / 255.0
        score = (inner_brightness * 100) + (ratio * 10)
        
        if score > best_score:
            best_score = score
            best_rect = (x, y, cw, ch)

    if best_rect is None:
        x, y, cw, ch = int(sw * 0.04), int(sh * 0.04), int(sw * 0.92), int(sh * 0.92)
        best_rect = (x, y, cw, ch)

    x, y, cw, ch = best_rect
    final_x = anchor_x + x
    final_y = anchor_y + y
    return image[final_y:final_y + ch, final_x:final_x + cw]

# --- ROUTE UTAMA API ---
@app.route('/api/ocr', methods=['POST'])
def process_ocr():
    if cnn_model is None:
        return jsonify({"success": False, "message": "Model AI belum terpasang di server Railway."}), 500

    data = request.get_json()
    if not data or 'image' Bone or 'image' not in data:
        return jsonify({"success": False, "message": "Gambar tidak dikirim."}), 400

    try:
        # Decode gambar base64 dari Hostinger
        encoded_data = data['image'].split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        source = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Preprocessing Gambar
        height, width = source.shape[:2]
        if width > 1400:
            scale = 1400 / float(width)
            source = cv2.resize(source, (1400, max(int(height * scale), 1)), interpolation=cv2.INTER_AREA)

        gray = grayscale(source)
        edges = prewitt_edges(gray)
        roi = find_digit_roi(source, gray, edges)
        
        # Trim ROI Border
        rh, rw = roi.shape[:2]
        rx, ry = max(int(rw * 0.02), 1), max(int(rh * 0.10), 1)
        rcw, rch = min(max(int(rw * 0.96), 1), rw - rx), min(max(int(rh * 0.80), 1), rh - ry)
        roi_trimmed = roi[ry:ry + rch, rx:rx + rcw]

        # Proses pembacaan 5 digit rata oleh CNN
        roi_gray = grayscale(roi_trimmed)
        th, tw = roi_gray.shape
        digit_width = tw // 5
        
        result_digits = ""
        confidences = []
        
        for i in range(5):
            start_x = i * digit_width
            end_x = (i + 1) * digit_width if i < 4 else tw
            digit_crop = roi_gray[:, start_x:end_x]
            
            resized_crop = cv2.resize(digit_crop, (64, 64), interpolation=cv2.INTER_AREA)
            img_array = resized_crop.astype('float32')
            img_array = np.expand_dims(np.expand_dims(img_array, axis=-1), axis=0)
            
            predictions = cnn_model.predict(img_array, verbose=0)
            result_digits += str(np.argmax(predictions[0]))
            confidences.append(np.max(predictions[0]) * 100)

        avg_confidence = sum(confidences) / len(confidences)
        
        # Konversi ROI ke base64 untuk dikirim balik ke UI Hostinger
        _, img_encoded = cv2.imencode('.jpg', roi_trimmed)
        roi_base64 = base64.b64encode(img_encoded).decode('utf-8')

        return jsonify({
            "success": True,
            "normalized_text": result_digits,
            "confidence": round(avg_confidence, 2),
            "roi_image": roi_base64,
            "engine": "railway-python-cnn-api"
        })

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

if __name__ == '__main__':
    # Jalankan server internal jika lokal test
    import base64
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
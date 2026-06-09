import sys
import json
import cv2
import numpy as np
import tensorflow as tf
import re
import os
import logging

# Matikan log yang tidak perlu
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
logging.getLogger("ppocr").setLevel(logging.ERROR)

from paddleocr import PaddleOCR

def segment_and_crop(gray_image, boxes):
    imgs = []
    for (x, y, w, h) in boxes[:6]: # Maksimal 6 digit
        digit = gray_image[y:y+h, x:x+w]
        hd, wd = digit.shape
        
        # ANTI-CRASH: Cegah error broadcast jika segmentasi terlalu besar
        if hd > 64 or wd > 64:
            scale = 64.0 / max(hd, wd)
            digit = cv2.resize(digit, (0, 0), fx=scale, fy=scale)
            hd, wd = digit.shape
            
        canvas = np.full((64, 64), 255, dtype=np.uint8)
        y_off = (64 - hd) // 2
        x_off = (64 - wd) // 2
        canvas[y_off:y_off+hd, x_off:x_off+wd] = digit
        imgs.append(canvas)
    return imgs

def main():
    try:
        image_path = sys.argv[1]
        model_path = sys.argv[2]
        
        # 1. OpenCV Preprocessing
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Crop aman di server (Fokus tengah meteran)
        h, w = gray.shape
        roi_color = img[int(h*0.2):int(h*0.8), int(w*0.1):int(w*0.9)]
        roi_gray = gray[int(h*0.2):int(h*0.8), int(w*0.1):int(w*0.9)]
        
        # 2. PaddleOCR (Pembaca Utama)
        ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        result_paddle = ocr.ocr(roi_color, cls=True)
        
        paddle_text = ""
        if result_paddle and result_paddle[0]:
            for line in result_paddle[0]:
                paddle_text += line[1][0]
        paddle_text = re.sub(r"\D", "", paddle_text) # Bersihkan dari huruf
        
        # 3. CNN Validator (Pengawas)
        model = tf.keras.models.load_model(model_path)
        
        # Segmentasi ROI dengan OpenCV untuk disuapkan ke CNN
        _, binary = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.dilate(binary, kernel, iterations=1)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        box_list = []
        rh, rw = binary.shape
        for c in contours:
            x, y, w_box, h_box = cv2.boundingRect(c)
            if h_box > rh * 0.20 and w_box > rw * 0.01:
                box_list.append((x, y, w_box, h_box))
        box_list.sort(key=lambda b: b[0])
        
        cnn_text = ""
        cnn_conf = 0.0
        
        if box_list:
            digit_images = segment_and_crop(roi_gray, box_list)
            batch = np.asarray(digit_images, dtype=np.float32) / 255.0
            batch = np.expand_dims(batch, axis=-1)
            
            preds = model.predict(batch, verbose=0)
            cnn_text = "".join([str(np.argmax(p)) for p in preds])
            cnn_conf = float(np.mean([np.max(p) * 100 for p in preds]))

        # 4. Logika Validasi Flow Nabil
        final_text = paddle_text
        engine_used = "paddleocr"
        
        # Jika Paddle kosong, atau CNN sangat yakin dan tidak mengulang 11111, CNN ambil alih
        if not paddle_text or (cnn_conf > 80 and not re.match(r"(\d)\1{3,}", cnn_text)):
            final_text = cnn_text
            engine_used = "cnn-validator"

        print(json.dumps({
            "success": True,
            "normalized_text": final_text,
            "confidence": round(cnn_conf, 2),
            "paddle_raw": paddle_text,
            "cnn_raw": cnn_text,
            "engine": engine_used
        }))

    except Exception as e:
        print(json.dumps({
            "success": False,
            "message": str(e)
        }))

if __name__ == "__main__":
    main()
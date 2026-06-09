import sys
import json
import cv2
import re
import os
import logging

# Matikan log default Paddle agar JSON tidak rusak
logging.getLogger("ppocr").setLevel(logging.ERROR)
os.environ.setdefault("KMP_WARNINGS", "0")

from paddleocr import PaddleOCR

def main():
    try:
        image_path = sys.argv[1]
        
        # Inisialisasi PaddleOCR (Hanya bahasa Inggris/Angka)
        ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        
        # Baca gambar dengan OpenCV
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Gambar tidak terbaca oleh server")

        # Proses gambar utuh, Paddle sangat pintar mencari area teks sendiri
        result = ocr.ocr(img, cls=True)
        
        final_text = ""
        conf_list = []

        if result and result[0]:
            for line in result[0]:
                text = line[1][0]
                conf = line[1][1]
                
                # Hanya ambil angka
                cleaned = re.sub(r"\D", "", text)
                if cleaned:
                    final_text += cleaned
                    conf_list.append(conf)

        if final_text:
            avg_conf = (sum(conf_list) / len(conf_list)) * 100
            print(json.dumps({
                "success": True,
                "normalized_text": final_text,
                "confidence": round(avg_conf, 2),
                "engine": "paddleocr-solo"
            }))
        else:
            print(json.dumps({
                "success": False,
                "message": "Tidak ada angka yang terdeteksi dengan jelas.",
                "engine": "paddleocr-solo"
            }))

    except Exception as e:
        print(json.dumps({"success": False, "message": str(e)}))

if __name__ == "__main__":
    main()
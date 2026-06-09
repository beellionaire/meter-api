import sys
import json
import re
import cv2
import warnings

# Matikan warning yang mengganggu output JSON
warnings.filterwarnings("ignore")

# PENTING: Matikan log default PaddleOCR agar tidak merusak output JSON
import logging
logging.getLogger("ppocr").setLevel(logging.ERROR)

from paddleocr import PaddleOCR

def main():
    try:
        image_path = sys.argv[1]
        
        # Inisialisasi PaddleOCR (Gunakan bahasa Inggris 'en' karena fokus pada angka)
        ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

        # Proses Gambar
        result = ocr.ocr(image_path, cls=True)
        
        final_text = ""
        confidence_list = []

        # Ekstraksi hasil
        if result and result[0]:
            for line in result[0]:
                # line format: [[koordinat_box], ('teks', confidence_score)]
                text = line[1][0]
                conf = line[1][1]
                
                # Filter hanya angka
                cleaned_text = re.sub(r"\D", "", text)
                if cleaned_text:
                    final_text += cleaned_text
                    confidence_list.append(conf)

        if final_text:
            avg_conf = (sum(confidence_list) / len(confidence_list)) * 100
            print(json.dumps({
                "success": True,
                "normalized_text": final_text,
                "confidence": round(avg_conf, 2),
                "engine": "paddleocr"
            }))
        else:
             print(json.dumps({
                "success": False,
                "message": "PaddleOCR tidak menemukan angka yang jelas.",
                "engine": "paddleocr"
            }))

    except Exception as e:
        print(json.dumps({
            "success": False,
            "message": str(e),
            "engine": "paddleocr"
        }))

if __name__ == "__main__":
    main()
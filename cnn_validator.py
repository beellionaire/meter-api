import sys, json, cv2, numpy as np, tensorflow as tf, pytesseract, re, base64

pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

def main():
    try:
        img_path = sys.argv[1]
        model_path = sys.argv[2]
        
        img = cv2.imread(img_path)
        if img is None: raise Exception("Gambar tidak terbaca")
        
        # Crop Otomatis (Ambil Tengah Saja)
        h, w = img.shape[:2]
        roi = img[int(h*0.3):int(h*0.7), int(w*0.2):int(w*0.8)]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Persiapkan ROI untuk dikirim kembali ke PHP agar kamu bisa lihat
        _, encoded_roi = cv2.imencode('.jpg', roi)
        roi_base64 = base64.b64encode(encoded_roi).decode('utf-8')
        
        # 1. Coba pakai CNN dulu (Paksa Resize ke 64x64 agar tidak broadcast error)
        model = tf.keras.models.load_model(model_path)
        img_resize = cv2.resize(gray, (64, 64))
        batch = np.expand_dims(np.expand_dims(img_resize/255.0, axis=0), axis=-1)
        pred = model.predict(batch, verbose=0)
        cnn_text = str(np.argmax(pred))
        conf = float(np.max(pred) * 100)
        
        final_text = cnn_text
        engine = "cnn"
        
        # 2. HYBRID LOGIC: Jika angka aneh atau kurang yakin, pakai Tesseract
        if conf < 70 or "1111" in cnn_text or "0000" in cnn_text:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            text = pytesseract.image_to_string(binary, config='--psm 7 -c tessedit_char_whitelist=0123456789')
            text = re.sub(r"\D", "", text)
            if text:
                final_text = text
                engine = "tesseract (cnn ragu)"
                conf = 95.0

        print(json.dumps({
            "success": True, 
            "normalized_text": final_text, 
            "confidence": round(conf, 2), 
            "engine": engine,
            "roi_image": roi_base64
        }))

    except Exception as e:
        print(json.dumps({"success": False, "message": str(e)}))

if __name__ == "__main__":
    main()
import argparse, json, os, re, sys, cv2, numpy as np, tensorflow as tf, pytesseract
from pathlib import Path
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

def preprocess_roi(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return gray, binary

def segment_digits(binary):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.dilate(binary, kernel, iterations=1)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = binary.shape[:2]
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h > height * 0.15 and w > width * 0.01:
            boxes.append((x, y, w, h))
    boxes.sort(key=lambda b: b[0])
    return boxes

def crop_digit(gray, box, image_size):
    x, y, w, h = box
    digit = gray[max(0, y):min(gray.shape[0], y+h), max(0, x):min(gray.shape[1], x+w)]
    canvas = np.full((64, 64), 255, dtype=np.uint8)
    h_d, w_d = digit.shape
    y_off, x_off = (64-h_d)//2, (64-w_d)//2
    canvas[y_off:y_off+h_d, x_off:x_off+w_d] = digit
    return cv2.resize(canvas, (image_size, image_size))

def predict_digits(model, gray, boxes, size):
    imgs = [crop_digit(gray, b, size) for b in boxes]
    batch = np.asarray(imgs, dtype=np.float32) / 255.0
    batch = np.expand_dims(batch, axis=-1)
    preds = model.predict(batch, verbose=0)
    digits = [str(int(np.argmax(p))) for p in preds]
    confs = [float(np.max(p)*100) for p in preds]
    return "".join(digits), confs

def run_tesseract(source_img):
    gray = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(binary, config='--psm 7 -c tessedit_char_whitelist=0123456789')
    return re.sub(r"\D", "", text)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path"); parser.add_argument("model_path"); parser.add_argument("previous_meter", type=int); parser.add_argument("processed_dir")
    args = parser.parse_args()
    
    source = cv2.imread(args.image_path)
    gray, binary = preprocess_roi(source)
    boxes = segment_digits(binary)
    model = tf.keras.models.load_model(args.model_path)
    
    cnn_digits, confs = predict_digits(model, gray, boxes, 64)
    avg_conf = np.mean(confs) if confs else 0
    
    # HYBRID LOGIC
    final_digits = cnn_digits
    engine = "cnn"
    if avg_conf < 80 or re.match(r"(\d)\1{3,}", cnn_digits):
        final_digits = run_tesseract(source)
        engine = "tesseract"

    print(json.dumps({"success": True, "normalized_text": final_digits, "confidence": round(float(avg_conf), 2), "engine": engine}))

if __name__ == "__main__":
    try: main()
    except Exception as e: print(json.dumps({"success": False, "message": str(e)}))
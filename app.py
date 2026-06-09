from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import base64
import os
import json

app = Flask(__name__)
CORS(app)

@app.route('/api/ocr', methods=['POST'])
def process_ocr():
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'success': False, 'message': 'Tidak ada gambar'})

        # Simpan sementara
        image_data = data['image'].split(',')[1]
        with open('temp.jpg', 'wb') as f:
            f.write(base64.b64decode(image_data))

        # Panggil CNN Validator dengan argumen lengkap
        cmd = [
            "python", "cnn_validator.py",
            "temp.jpg",
            "models/cnn_digit_meter.keras",
            str(data.get('previous_meter', 0)),
            "processed"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if os.path.exists('temp.jpg'): os.remove('temp.jpg')
        
        # Kembalikan output dari Python
        return jsonify(json.loads(result.stdout))
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
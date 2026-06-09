from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import base64
import os
import json

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def index():
    return jsonify({"status": "API PaddleOCR Aktif!"})

@app.route('/api/ocr', methods=['POST'])
def process_ocr():
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'success': False, 'message': 'Tidak ada gambar yang dikirim'})

        # Decode Base64 dari PHP
        image_data = data['image'].split(',')[1]
        with open('temp.jpg', 'wb') as f:
            f.write(base64.b64decode(image_data))

        # Eksekusi Skrip PaddleOCR
        cmd = ["python", "paddle_validator.py", "temp.jpg"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Hapus file sementara
        if os.path.exists('temp.jpg'):
            os.remove('temp.jpg')

        # Kembalikan output JSON
        return jsonify(json.loads(result.stdout))

    except Exception as e:
        return jsonify({'success': False, 'message': f'Error Server: {str(e)}'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
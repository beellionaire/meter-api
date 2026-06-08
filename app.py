from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import base64
import os
import json

app = Flask(__name__)
CORS(app) # Mengizinkan PHP (Shared Hosting) memanggil API ini

@app.route('/', methods=['GET'])
def index():
    return jsonify({"status": "API OCR Meteran Air Aktif!"})

@app.route('/api/ocr', methods=['POST'])
def process_ocr():
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'success': False, 'message': 'Tidak ada gambar yang dikirim'})

        # 1. Decode foto Base64 dari PHP dan simpan jadi temp.jpg
        image_data = data['image'].split(',')[1]
        with open('temp.jpg', 'wb') as f:
            f.write(base64.b64decode(image_data))

        previous_meter = data.get('previous_meter', 0)

        # 2. Panggil script cnn_validator.py milikmu persis seperti di terminal
        cmd = [
            "python", "cnn_validator.py",
            "temp.jpg",                     # Gambar input
            "models/cnn_digit_meter.keras", # Path model (sesuaikan ekstensi keras/h5)
            str(previous_meter),            # Meter sebelumnya
            "processed"                     # Folder output ROI
        ]

        # 3. Jalankan dan tangkap output JSON-nya
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # 4. Hapus foto sementara
        if os.path.exists('temp.jpg'):
            os.remove('temp.jpg')

        # 5. Kembalikan hasil dari Python ke PHP
        output_json = json.loads(result.stdout)
        return jsonify(output_json)

    except Exception as e:
        return jsonify({'success': False, 'message': f'Error Server: {str(e)}'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
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
            return jsonify({'success': False, 'message': 'Gambar kosong'})

        # Decode gambar
        image_data = data['image'].split(',')[1]
        with open('temp.jpg', 'wb') as f:
            f.write(base64.b64decode(image_data))

        # Panggil Paddle API
        cmd = ["python", "paddle_api.py", "temp.jpg"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if os.path.exists('temp.jpg'):
            os.remove('temp.jpg')

        return jsonify(json.loads(result.stdout))

    except Exception as e:
        return jsonify({'success': False, 'message': f'Internal Server Error: {str(e)}'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
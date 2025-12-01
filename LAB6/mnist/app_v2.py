"""
MNIST Digit Classifier - Version 2.0
Dark theme with enhanced UI - demonstrates Terraform reapply
"""

from flask import Flask, request, jsonify, render_template_string
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

# Version info
VERSION = "2.0"

# Load the trained model
print("Loading model...")
model = tf.keras.models.load_model('model/mnist_model.h5')
print("Model loaded successfully!")

# HTML template - V2 DARK THEME
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>MNIST Classifier v2.0</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            padding: 2rem;
            color: #e0e0e0;
        }
        .container {
            max-width: 700px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            margin-bottom: 2rem;
        }
        h1 {
            font-size: 2.5rem;
            background: linear-gradient(90deg, #e94560, #ff6b6b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        .version-badge {
            display: inline-block;
            background: linear-gradient(90deg, #e94560, #ff6b6b);
            color: white;
            padding: 0.3rem 1rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: bold;
        }
        .card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 1.5rem;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .info-box {
            background: rgba(233, 69, 96, 0.1);
            border-left: 4px solid #e94560;
            padding: 1rem;
            border-radius: 0 8px 8px 0;
            margin-bottom: 1.5rem;
        }
        .info-box ul {
            padding-left: 1.5rem;
            margin-top: 0.5rem;
        }
        .info-box li {
            margin: 0.3rem 0;
            color: #ccc;
        }
        .upload-area {
            border: 2px dashed rgba(233, 69, 96, 0.5);
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
            transition: all 0.3s;
            cursor: pointer;
            margin-bottom: 1rem;
        }
        .upload-area:hover {
            border-color: #e94560;
            background: rgba(233, 69, 96, 0.1);
        }
        .upload-area.has-file {
            border-color: #4ecdc4;
            background: rgba(78, 205, 196, 0.1);
        }
        input[type="file"] { display: none; }
        #preview {
            max-width: 200px;
            max-height: 200px;
            margin: 1rem auto;
            display: block;
            border-radius: 8px;
            border: 2px solid #e94560;
        }
        button {
            width: 100%;
            padding: 1rem 2rem;
            background: linear-gradient(90deg, #e94560, #ff6b6b);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1.1rem;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(233, 69, 96, 0.4);
        }
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        .result {
            text-align: center;
            margin-top: 1.5rem;
            display: none;
        }
        .result.show { display: block; }
        .prediction-number {
            font-size: 6rem;
            font-weight: bold;
            background: linear-gradient(90deg, #4ecdc4, #44a08d);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            line-height: 1;
        }
        .confidence-bar {
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            height: 10px;
            margin: 1rem 0;
            overflow: hidden;
        }
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #4ecdc4, #44a08d);
            border-radius: 10px;
            transition: width 0.5s ease;
        }
        .confidence-text {
            color: #4ecdc4;
            font-size: 1.2rem;
        }
        .footer {
            text-align: center;
            margin-top: 2rem;
            color: #666;
        }
        .footer span {
            display: inline-block;
            background: rgba(255,255,255,0.1);
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-size: 0.85rem;
        }
        .loading {
            display: none;
            text-align: center;
            padding: 2rem;
        }
        .loading.show { display: block; }
        .spinner {
            width: 50px;
            height: 50px;
            border: 4px solid rgba(233, 69, 96, 0.3);
            border-top-color: #e94560;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔢 MNIST Classifier</h1>
            <span class="version-badge">v2.0 - DARK MODE</span>
        </div>
        
        <div class="card">
            <div class="info-box">
                <strong>📌 What's New in v2.0:</strong>
                <ul>
                    <li>Dark theme UI</li>
                    <li>Enhanced prediction visualization</li>
                    <li>Deployed via Terraform reapply</li>
                </ul>
            </div>
            
            <div class="upload-area" id="uploadArea" onclick="document.getElementById('imageInput').click()">
                <p style="font-size: 2rem; margin-bottom: 0.5rem;">📷</p>
                <p>Click to upload a digit image</p>
                <p style="color: #888; font-size: 0.85rem; margin-top: 0.5rem;">PNG, JPG, or GIF</p>
                <input type="file" id="imageInput" accept="image/*">
            </div>
            
            <img id="preview" style="display:none;">
            
            <button onclick="predict()" id="predictBtn">🎯 Predict Digit</button>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p style="margin-top: 1rem;">Analyzing...</p>
            </div>
            
            <div class="result" id="result">
                <p style="color: #888;">Predicted Digit:</p>
                <div class="prediction-number" id="predictionNumber">-</div>
                <div class="confidence-bar">
                    <div class="confidence-fill" id="confidenceFill" style="width: 0%"></div>
                </div>
                <p class="confidence-text">Confidence: <span id="confidenceText">0%</span></p>
            </div>
        </div>
        
        <div class="footer">
            <span>MLOps LAB6 | Terraform Deployment | TensorFlow CNN</span>
        </div>
    </div>

    <script>
        document.getElementById('imageInput').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    const preview = document.getElementById('preview');
                    preview.src = e.target.result;
                    preview.style.display = 'block';
                    document.getElementById('uploadArea').classList.add('has-file');
                    document.getElementById('result').classList.remove('show');
                }
                reader.readAsDataURL(file);
            }
        });

        async function predict() {
            const fileInput = document.getElementById('imageInput');
            const loading = document.getElementById('loading');
            const result = document.getElementById('result');
            const btn = document.getElementById('predictBtn');
            
            if (!fileInput.files[0]) {
                alert('Please select an image first!');
                return;
            }

            btn.disabled = true;
            loading.classList.add('show');
            result.classList.remove('show');

            const formData = new FormData();
            formData.append('image', fileInput.files[0]);

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();
                
                if (data.error) {
                    alert('Error: ' + data.error);
                } else {
                    document.getElementById('predictionNumber').textContent = data.predicted_digit;
                    const confidence = (data.confidence * 100).toFixed(1);
                    document.getElementById('confidenceText').textContent = confidence + '%';
                    document.getElementById('confidenceFill').style.width = confidence + '%';
                    result.classList.add('show');
                }
            } catch (error) {
                alert('Error: ' + error.message);
            } finally {
                loading.classList.remove('show');
                btn.disabled = false;
            }
        }
    </script>
</body>
</html>
"""

def preprocess_image(image_bytes):
    """Preprocess the uploaded image for model prediction"""
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert('L')
    image = image.resize((28, 28))
    image_array = np.array(image)
    image_array = image_array.astype('float32') / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)
    return image_array

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'version': VERSION,
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        image_bytes = file.read()
        processed_image = preprocess_image(image_bytes)
        
        predictions = model.predict(processed_image, verbose=0)
        predicted_digit = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        
        return jsonify({
            'predicted_digit': predicted_digit,
            'confidence': confidence,
            'version': VERSION
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)


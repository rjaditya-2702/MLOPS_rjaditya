from flask import Flask, request, jsonify, render_template_string
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load the trained model
print("Loading model...")
model = tf.keras.models.load_model('model/mnist_model.h5')
print("Model loaded successfully!")

# HTML template for testing
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>MNIST Digit Classifier</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .upload-section {
            margin: 30px 0;
            text-align: center;
        }
        input[type="file"] {
            margin: 20px 0;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background-color: #45a049;
        }
        #result {
            margin-top: 30px;
            padding: 20px;
            background-color: #e8f5e9;
            border-radius: 5px;
            display: none;
        }
        #preview {
            margin: 20px 0;
            text-align: center;
        }
        #preview img {
            border: 2px solid #ddd;
            border-radius: 5px;
            max-width: 200px;
        }
        .info {
            background-color: #e3f2fd;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }
        .prediction {
            font-size: 48px;
            font-weight: bold;
            color: #4CAF50;
            text-align: center;
            margin: 20px 0;
        }
        .confidence {
            text-align: center;
            color: #666;
            font-size: 18px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔢 MNIST Digit Classifier</h1>
        <div class="info">
            <strong>Instructions:</strong>
            <ul>
                <li>Upload an image of a handwritten digit (0-9)</li>
                <li>The image should be grayscale and contain a single digit</li>
                <li>Best results with digits on white/light background</li>
            </ul>
        </div>
        
        <div class="upload-section">
            <input type="file" id="imageInput" accept="image/*">
            <br>
            <button onclick="predict()">Predict Digit</button>
        </div>
        
        <div id="preview"></div>
        <div id="result"></div>
    </div>

    <script>
        document.getElementById('imageInput').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    document.getElementById('preview').innerHTML = 
                        '<img src="' + e.target.result + '" alt="Preview">';
                }
                reader.readAsDataURL(file);
            }
        });

        async function predict() {
            const fileInput = document.getElementById('imageInput');
            const resultDiv = document.getElementById('result');
            
            if (!fileInput.files[0]) {
                alert('Please select an image first!');
                return;
            }

            const formData = new FormData();
            formData.append('image', fileInput.files[0]);

            try {
                resultDiv.style.display = 'none';
                resultDiv.innerHTML = '<p>Predicting...</p>';
                resultDiv.style.display = 'block';

                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();
                
                if (data.error) {
                    resultDiv.innerHTML = '<p style="color: red;">Error: ' + data.error + '</p>';
                } else {
                    resultDiv.innerHTML = `
                        <h2>Prediction Result</h2>
                        <div class="prediction">${data.predicted_digit}</div>
                        <div class="confidence">Confidence: ${(data.confidence * 100).toFixed(2)}%</div>
                        <hr>
                        <p><strong>All Probabilities:</strong></p>
                        <pre>${JSON.stringify(data.all_probabilities, null, 2)}</pre>
                    `;
                }
                resultDiv.style.display = 'block';
            } catch (error) {
                resultDiv.innerHTML = '<p style="color: red;">Error: ' + error.message + '</p>';
                resultDiv.style.display = 'block';
            }
        }
    </script>
</body>
</html>
"""

def preprocess_image(image_bytes):
    """Preprocess the uploaded image for model prediction"""
    # Load image
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to grayscale
    image = image.convert('L')
    
    # Resize to 28x28
    image = image.resize((28, 28))
    
    # Convert to numpy array
    image_array = np.array(image)
    
    # Normalize pixel values
    image_array = image_array.astype('float32') / 255.0
    
    # Reshape for model input
    image_array = image_array.reshape(1, 28, 28, 1)
    
    return image_array

@app.route('/')
def home():
    """Serve the web interface"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict digit from uploaded image"""
    try:
        # Check if image was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Read and preprocess image
        image_bytes = file.read()
        processed_image = preprocess_image(image_bytes)
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        predicted_digit = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        
        # Get all probabilities
        all_probs = {str(i): float(predictions[0][i]) for i in range(10)}
        
        return jsonify({
            'predicted_digit': predicted_digit,
            'confidence': confidence,
            'all_probabilities': all_probs
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_array', methods=['POST'])
def predict_array():
    """Predict digit from raw numpy array (for API testing)"""
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'No image array provided'}), 400
        
        # Convert to numpy array and reshape
        image_array = np.array(data['image'], dtype='float32')
        
        # Ensure correct shape
        if image_array.shape != (28, 28):
            image_array = image_array.reshape(28, 28)
        
        # Normalize if needed
        if image_array.max() > 1.0:
            image_array = image_array / 255.0
        
        # Reshape for model
        image_array = image_array.reshape(1, 28, 28, 1)
        
        # Make prediction
        predictions = model.predict(image_array, verbose=0)
        predicted_digit = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        
        return jsonify({
            'predicted_digit': predicted_digit,
            'confidence': confidence
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
# MNIST Digit Classifier - Docker Deployment

A containerized machine learning application that trains a CNN on MNIST and serves predictions via Flask API.

## Project Structure

```
LAB5/
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
├── src/
│   ├── train.py
│   ├── app.py
└── model/
    └── mnist_model.h5
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Model
```bash
python src/train.py
```

### 3. Run Locally (Optional)
```bash
python src/app.py
# Visit http://localhost:5001
```

### 4. Deploy with Docker

**Option A: Using Docker Compose (Easiest)**
```bash
docker-compose up --build
```

**Option B: Using Docker Commands**
```bash
# Build
docker build -t mnist-classifier .

# Run with volume mount
docker run -d -p 8080:5000 -v $(pwd)/model:/app/model --name mnist_app mnist-classifier
```

### 5. Test
```bash
# Health check
curl http://localhost:8080/health

# Web interface
open http://localhost:8080

# Run tests
python src/test_api.py
```

## Requirements

- Python 3.10+
- Docker & Docker Compose
- 2GB+ RAM

## Docker Commands

```bash
# Build image
docker build -t mnist-classifier .

# Run container
docker run -d -p 8080:5000 -v $(pwd)/model:/app/model --name mnist_app mnist-classifier

# View logs
docker logs -f mnist_app

# Stop container
docker stop mnist_app

# Remove container
docker rm mnist_app

# Clean up
docker-compose down
```

## Model Details

- **Architecture**: CNN (3 Conv layers + 2 Dense layers)
- **Dataset**: MNIST (60,000 training images)
- **Training Time**: 2-3 minutes on CPU

## API Endpoints

### GET `/`
Web interface for uploading images

### GET `/health`
```json
{"status": "healthy", "model_loaded": true}
```

### POST `/predict`
Upload image file, returns prediction
```json
{
  "predicted_digit": 5,
  "confidence": 0.9987,
  "all_probabilities": {...}
}
```

### POST `/predict_array`
Send 28x28 array, returns prediction

## Files Description

| File | Purpose |
|------|---------|
| `src/train.py` | Train MNIST CNN model |
| `src/app.py` | Flask API for predictions |
| `src/test_api.py` | API testing script |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Docker container config |
| `docker-compose.yml` | Multi-container setup |
| `.dockerignore` | Exclude files from build |
| `.gitignore` | Exclude files from Git |

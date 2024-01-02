# Flask Deep Learning Model Deployment

## Project Overview

This Flask project demonstrates the deployment of two deep learning models for different tasks, namely image classification and text classification. The goal is to enable server-side inference through HTTP requests.

## Team Members
- Motassim Ahmed Taha

## Tasks and Model Selection

### Task 1: Image Classification
- **Selected Model:** ResNet50
- **Motivation:** ResNet50 is a powerful convolutional neural network (CNN) architecture widely used for image classification. It has shown excellent performance on various image datasets.

### Task 2: Text Classification
- **Selected Model:** BERT (Bidirectional Encoder Representations from Transformers)
- **Motivation:** BERT is a state-of-the-art transformer-based model known for its superior performance in natural language processing tasks. It excels in capturing contextual information and relationships within text.

## Model Integration

We ensured correct usage of the selected models by following these steps:

1. **Model Selection:** We chose models that are pre-trained on large datasets and suitable for the respective tasks.
2. **Testing:** We performed testing on sample data to ensure the models provide accurate predictions.
3. **Compatibility:** Ensured that the models are compatible with the libraries and frameworks used for deployment (e.g., PyTorch, Hugging Face).

## Deployment

The models are deployed using Flask, PyTorch, and Hugging Face. The server enables inference through POST HTTP requests and returns results in JSON format.

### Endpoints

- **Image Classification Endpoint:** `/predict_image`
- **Text Classification Endpoint:** `/predict_text`

### Example Usage

#### Image Classification
```bash
curl -X POST -H "Content-Type: application/json" -d '{"image_url": "https://example.com/image.jpg"}' http://localhost:5000/predict_image
```

### Web Interface

We have developed a user-friendly web interface to illustrate the use of the server. The interface includes the following routes:

- /: Home page with project overview and instructions.

- /image_classification: Interface for image classification with an option to upload an image or provide a URL.

- /text_classification: Interface for text classification with a text input box.

Feel free to explore the web interface and experience the seamless integration of deep learning models for image and text classification.

### Getting Started

To run the project locally, follow these steps:

1. Clone the repository:
```bash
git clone https://github.com/yourusername/flask-deep-learning.git
cd flask-deep-learning
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Access the web interface at http://localhost:5000 and start testing the models.

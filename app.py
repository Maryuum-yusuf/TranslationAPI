from flask import Flask, request, jsonify
from transformers import MarianTokenizer, TFMarianMTModel
import tensorflow as tf
import numpy as np

model_path = "./amiin_model"

tokenizer = MarianTokenizer.from_pretrained(model_path)
model = TFMarianMTModel.from_pretrained(model_path)

app = Flask(__name__)

@app.route("/")
def home():
    return "Translation API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("input", "")

    if not text:
        return jsonify({"error": "No input provided"}), 400

    # Tokenize
    inputs = tokenizer([text], return_tensors="tf", padding=True)
    
    # Translate
    outputs = model.generate(**inputs)
    translation = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return jsonify({"translation": translation[0]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

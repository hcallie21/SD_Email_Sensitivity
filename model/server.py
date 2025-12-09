from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re

app = Flask(__name__)
CORS(app)  

# CONFIGURATION
MODEL_PATH = "./saved_model"
OVERALL_THRESHOLD = 0.90
SENTENCE_THRESHOLD = 0.90
EMAIL_REGEX = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
PHONE_REGEX = r'\b(?:\+?1[-.]?)?\(?[0-9]{3}\)?[-.]?[0-9]{3}[-.]?[0-9]{4}\b'
SSN_REGEX = r'\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b'
NAME_REGEX = r'(?:[Mm]y name is|[Nn]ame is|[Nn]ame:)\s+[A-Z][a-z]+' 
SAFE_GREETINGS = [
    r'^hi\b', r'^hello\b', r'^dear\b', r'^thanks\b', r'^thank you\b', 
    r'^best\b', r'^regards\b', r'^sincerely\b', r'^hope you are\b', r'^stay safe\b'
]
print("Loading model from ./saved_model...")
print("Loading model from ./saved_model...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

def split_into_sentences(text):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s|\n+', text)
    return [s.strip() for s in sentences if s.strip()]

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        score = probs[0][1].item()
        prediction = 1 if score >= SENTENCE_THRESHOLD else 0
        return prediction, score

def get_highlight_spans(text):
    spans = []
    for regex in [EMAIL_REGEX, PHONE_REGEX, SSN_REGEX, NAME_REGEX]:
        matches = re.findall(regex, text)
        spans.extend(matches)
    
    return list(set([s for s in spans if s]))

def is_safe_greeting(text):
    text_lower = text.lower()
    for pattern in SAFE_GREETINGS:
        if re.search(pattern, text_lower):
            return True
    return False

@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.json
    text = data.get('text', '')
    
    sentences = split_into_sentences(text)
    results = []
    
    for sentence in sentences:
        prediction, confidence = predict(sentence)
        is_sensitive = (prediction == 1)
        spans = get_highlight_spans(sentence)
        
        if len(spans) > 0:
            is_sensitive = True
            confidence = max(confidence, 0.95)

        if is_sensitive and len(spans) == 0:
            if is_safe_greeting(sentence):
                is_sensitive = False
            
        pii_count = len(spans) 
        if is_sensitive and pii_count == 0:
            pii_count = 1 

        results.append({
            "text": sentence,
            "prediction": 1 if is_sensitive else 0,
            "confidence": float(confidence),
            "pii_count": pii_count
        })
    
    return jsonify({"sentences": results})

if __name__ == '__main__':
    app.run(debug=True, port=5000)

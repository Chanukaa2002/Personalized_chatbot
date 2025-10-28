from flask import Flask, request, jsonify
from flask_cors import CORS
import json, os, random, time
from sentence_transformers import SentenceTransformer, util
import torch
from sklearn.preprocessing import LabelEncoder
import pickle
from rapidfuzz import process, fuzz

app = Flask(__name__)
CORS(app)

CONFIDENCE_THRESHOLD = float(os.getenv('THRESHOLD', '0.55'))
FUZZY_THRESHOLD = 75 

# Load intents
with open('data/intents.json', encoding='utf-8') as f:
    intents_data = json.load(f)

try:
    vectorizer = pickle.load(open('./model/vectorizer.pkl', 'rb'))
    encoder = pickle.load(open('./model/label_encoder.pkl', 'rb'))
    model = None
    print("✅ Vectorizer and encoder loaded!")
except Exception as e:
    print("❌ Error loading vectorizer/encoder:", e)
    vectorizer = None
    encoder = None
    model = None

# embed_model = SentenceTransformer('all-MiniLM-L6-v2')
embed_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

print("⚡ Warming up SentenceTransformer model...")
_ = embed_model.encode("Hello", convert_to_tensor=True)
print("✅ SentenceTransformer ready!")

patterns = []
pattern_tags = []
for intent in intents_data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern.lower())
        pattern_tags.append(intent['tag'])
pattern_embeddings = embed_model.encode(patterns, convert_to_tensor=True)
pattern_embeddings.requires_grad = False

def preprocess(text):
    text = text.lower()
    text = text.replace("tnx", "thanks").replace("thx", "thanks")
    text = text.replace("helo", "hello").replace("hii", "hi")
    text = text.replace("wht", "what").replace("abt", "about")
    return text

@app.route("/chat", methods=["POST"])
def chat():
    random.seed(time.time())
    user_msg = request.json.get('message', '')
    if not user_msg:
        return jsonify({"reply": "Please send a message in the 'message' field."}), 400

    user_msg_clean = preprocess(user_msg)

    for intent_data in intents_data['intents']:
        for pattern in intent_data['patterns']:
            if user_msg_clean == pattern.lower():
                response = random.choice(intent_data.get('responses', []))
                return jsonify({"reply": response, "intent": intent_data['tag'], "confidence": 1.0})

    result = process.extractOne(user_msg_clean, patterns, scorer=fuzz.WRatio)
    if result:
        if isinstance(result, tuple):
            match, score, _ = result
        else:
            match = result['match']
            score = result['score']

        if score >= FUZZY_THRESHOLD:
            intent = pattern_tags[patterns.index(match)]
            intent_responses = next(
                (i.get('responses', []) for i in intents_data['intents'] if i['tag'] == intent),
                []
            )
            if intent_responses:
                response = random.choice(intent_responses)
                return jsonify({"reply": response, "intent": intent, "confidence": round(score/100, 2)})


    user_embedding = embed_model.encode(user_msg_clean, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(user_embedding, pattern_embeddings)

    top_scores, top_indices = torch.topk(cos_scores, k=min(3, len(patterns)))
    responses = []
    primary_intent = None
    primary_score = 0.0
    for score, idx in zip(top_scores[0], top_indices[0]):
        if float(score) >= CONFIDENCE_THRESHOLD:
            tag = pattern_tags[idx]
            intent_responses = next(
                (i.get('responses', []) for i in intents_data['intents'] if i['tag'] == tag),
                []
            )
            if intent_responses:
                responses.append(random.choice(intent_responses))
                if not primary_intent:
                    primary_intent = tag
                    primary_score = float(score)

    if responses:
        response = " ".join(responses)
        intent = primary_intent
        top_score = primary_score
    else:
        os.makedirs("data", exist_ok=True)
        with open("data/unanswered_logs.json", "a", encoding="utf-8") as f:
            json.dump({"message": user_msg}, f)
            f.write("\n")
        fallback = (
            "I only answer questions about Chanuka Dilshan — projects, skills, education, and related topics. "
            "If you have a question about him, please ask differently or check his website: https://www.chanukadilshan.live/"
        )
        return jsonify({"reply": fallback, "intent": "unknown", "confidence": 0.0})

    if model and vectorizer and encoder:
        try:
            X_new = vectorizer.transform([user_msg_clean])
            y_new = encoder.transform([intent])
            model.fit(X_new.toarray(), y_new, epochs=1, batch_size=1, verbose=0)
        except:
            pass

    return jsonify({"reply": response, "intent": intent, "confidence": round(top_score, 2)})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)

from flask import Flask, request, jsonify
from flask_cors import CORS
from LLM import load_model, generate

app = Flask(__name__)
CORS(app)  # allow frontend access

# --- Initialize model ---
model = load_model()  # automatically loads model.pth

@app.route("/generate", methods=["POST"])
def generate_text():
    try:
        data = request.get_json()
        prompt = data.get("prompt", "").lower().strip()
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400

        result = generate(model, prime_str=prompt, predict_len=150)
        return jsonify({"response": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

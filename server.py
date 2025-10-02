from flask import Flask, request, jsonify
import traceback

# Import your existing mini-report function
# (adjust the function name if it's not actually called generate_report)
from pdf_generator.generate_mini_report import generate_report 

app = Flask(__name__)

@app.route("/generate", methods=["POST"])
def generate():
    try:
        payload = request.json or {}
        pdf_url = generate_report(payload)
        return jsonify({"status": "ok", "pdf_url": pdf_url})
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "trace": traceback.format_exc()
        }), 500

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Cloud Run gives us PORT
    app.run(host="0.0.0.0", port=port)
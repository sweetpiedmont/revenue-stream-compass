from flask import Flask, request, jsonify
import traceback
from pdf_generator.generate_mini_report import generate_report

app = Flask(__name__)

@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json(silent=True) or {}

        # Call your existing function
        pdf_url = generate_report(data)

        return jsonify({
            "status": "ok",
            "pdf_url": pdf_url,
            "input": data
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "trace": traceback.format_exc()
        }), 500
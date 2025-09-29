from flask import Flask, request, jsonify
import traceback

# Import your existing mini-report function
# (adjust the function name if it's not actually called generate_report)
from generate_mini_report import generate_report  

app = Flask(__name__)

@app.route("/", methods=["POST"])
def main():
    try:
        payload = request.json or {}
        # Example: assume payload has {"user_scores": {...}, "user_email": "..."}
        pdf_url = generate_report(payload)

        return jsonify({"status": "ok", "pdf_url": pdf_url})

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "trace": traceback.format_exc()
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

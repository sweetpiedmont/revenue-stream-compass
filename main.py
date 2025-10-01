from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/generate", methods=["POST"])
def generate():
    # Grab JSON body from Zapier
    data = request.get_json(silent=True) or {}
    return jsonify({
        "message": "✅ Mini report service is alive!",
        "you_sent": data
    })
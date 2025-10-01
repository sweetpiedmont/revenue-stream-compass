from flask import Flask, request, jsonify
import traceback
from pdf_generator.generate_mini_report import generate_report
from google.cloud import storage
from datetime import timedelta
import os

app = Flask(__name__)

def upload_and_get_signed_url(filename, blob_name):
    client = storage.Client()
    bucket = client.bucket("rsc-mini-reports")
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(filename)

    # Signed URL (valid for 7 days)
    url = blob.generate_signed_url(expiration=timedelta(days=7), method="GET")
    return url

@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json(silent=True) or {}

        # 1. Generate the report (assume this returns a local file path)
        local_pdf_path = generate_report(data)

        # 2. Derive blob name (e.g., mini-report-<record_id>.pdf)
        record_id = data.get("record_id", "unknown")
        blob_name = f"mini-report-{record_id}.pdf"

        # 3. Upload + get signed URL
        signed_url = upload_and_get_signed_url(local_pdf_path, blob_name)

        # 4. Return JSON with signed URL
        return jsonify({
            "status": "ok",
            "pdf_url": signed_url,
            "input": data
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "trace": traceback.format_exc()
        }), 500
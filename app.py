import os
import cv2
from flask import Flask, render_template, flash, request, send_file
from werkzeug.utils import secure_filename
from utils import embed_message, extract_message
from metrics import evaluate_performance

app = Flask(__name__)
app.secret_key = "your_secret_key"

UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

@app.route("/")
def home():
    return render_template("index.html", active_tab="home")


# ---------------- Sender ----------------
@app.route("/encode", methods=["POST"])
def encode():
    print(">>> /encode route triggered")

    image_file = request.files.get("image")
    message = request.form.get("message", "")
    key = request.form.get("key", "")

    if not image_file or not message:
        flash("Image and message are required", "danger")
        return render_template("index.html", active_tab="sender")

    filename = secure_filename(image_file.filename)
    in_path = os.path.join(UPLOAD_FOLDER, filename)
    image_file.save(in_path)

    img = cv2.imread(in_path)
    if img is None:
        flash("Error: could not read uploaded image.", "danger")
        return render_template("index.html", active_tab="sender")

    try:
        stego = embed_message(img, message, key)
    except Exception as e:
        flash(f"Embedding error: {e}", "danger")
        return render_template("index.html", active_tab="sender")

    out_name = f"stego_{filename}"
    out_path = os.path.join(OUTPUT_FOLDER, out_name)
    success = cv2.imwrite(out_path, stego)
    if not success:
        flash("Error: could not save stego image.", "danger")
        return render_template("index.html", active_tab="sender")

    flash("Stego image generated successfully.", "success")
    return render_template(
        "index.html",
        stego_image=f"outputs/{out_name}",
        download_link=out_name,
        active_tab="sender"
    )


@app.route("/download/<filename>")
def download_file(filename):
    path = os.path.join(OUTPUT_FOLDER, filename)
    return send_file(path, as_attachment=True)


# ---------------- Receiver ----------------
@app.route("/decode", methods=["POST"])
def decode():
    stego_file = request.files.get("stego_image")
    key = request.form.get("key", "")

    if not stego_file:
        flash("Stego image is required", "danger")
        return render_template("index.html", active_tab="receiver")

    filename = secure_filename(stego_file.filename)
    in_path = os.path.join(UPLOAD_FOLDER, filename)
    stego_file.save(in_path)

    stego = cv2.imread(in_path)
    if stego is None:
        flash("Error: could not read stego image.", "danger")
        return render_template("index.html", active_tab="receiver")

    try:
        message = extract_message(stego, key)
    except Exception as e:
        flash(f"Extraction error: {e}", "danger")
        return render_template("index.html", active_tab="receiver")

    flash("Message extracted successfully.", "success")
    return render_template(
        "index.html",
        extracted_message=message,
        active_tab="receiver"
    )


# ---------------- Performance ----------------
@app.route("/performance", methods=["POST"])
def performance():
    orig_file = request.files.get("original_image")
    stego_file = request.files.get("stego_image")

    if not orig_file or not stego_file:
        flash("Please upload both original and stego images.", "danger")
        return render_template("index.html", active_tab="performance")

    orig_filename = secure_filename(orig_file.filename)
    stego_filename = secure_filename(stego_file.filename)
    orig_path = os.path.join(UPLOAD_FOLDER, orig_filename)
    stego_path = os.path.join(UPLOAD_FOLDER, stego_filename)
    orig_file.save(orig_path)
    stego_file.save(stego_path)

    original = cv2.imread(orig_path)
    stego = cv2.imread(stego_path)
    if original is None or stego is None:
        flash("Error: could not read one of the images.", "danger")
        return render_template("index.html", active_tab="performance")

    metrics = evaluate_performance(original, stego)

    return render_template(
        "index.html",
        metrics=metrics,
        active_tab="performance",
        original_image=f"uploads/{orig_filename}",
        stego_image=f"uploads/{stego_filename}"
    )


if __name__ == "__main__":
    app.run(debug=True)

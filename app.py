import os
import base64
import requests
import torch
import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
import torchvision.transforms as transforms
from openai import OpenAI

from model import SkinModel
from utils.gradcam import GradCAM

# ---------------- APP ----------------
app = Flask(__name__, template_folder="templates", static_folder="static")

# ---------------- OPENAI ----------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

ELEVENLABS_API_KEY = "sk_c8168fb5ac1b9fbfee4b35326b6d8c195842d3329c0195e2"
VOICE_ID = "21m00Tcm4TlvDq8ikWAM"

PROMPT_ID = "pmpt_69dbac32d5c08194a4e0f899b1b5625705ceec2a9a9eb28e"
PROMPT_VERSION = "2"

# ---------------- MODEL ----------------
MODEL_PATH = "best_model.pth"
NUM_CLASSES = 14

CLASS_NAMES = [
    "Actinic keratoses","Basal cell carcinoma","Benign keratosis-like lesions",
    "Chickenpox","Cowpox","Dermatofibroma","HFMD","Healthy","Measles",
    "Melanocytic nevi","Melanoma","Monkeypox","Squamous cell carcinoma","Vascular lesions"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SkinModel(num_classes=NUM_CLASSES).to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

cam = None

def split_text(text, max_len=200):
    sentences = text.split(". ")
    chunks = []
    current = ""

    for s in sentences:
        if len(current) + len(s) < max_len:
            current += s + ". "
        else:
            chunks.append(current.strip())
            current = s + ". "

    if current:
        chunks.append(current.strip())

    return chunks

# ---------------- LOAD MODEL ----------------
def load_model():
    global cam

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    state = checkpoint.get("model") or checkpoint.get("state_dict") or checkpoint

    cleaned = {}
    for k, v in state.items():
        cleaned[k.replace("module.", "").replace("model.", "")] = v

    model.load_state_dict(cleaned, strict=False)
    model.eval()

    cam = GradCAM(model, model.backbone.features[-1])
    print("Model + GradCAM ready")

def warmup():
    x = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        _ = model(x)

# ---------------- IMAGE ----------------
def encode(img):
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf).decode()

def run(file):
    img = Image.open(file).convert("RGB").resize((224, 224))
    arr = np.array(img)

    t = transform(img).unsqueeze(0).to(device)
    t = t.clone().detach().requires_grad_(True)

    out = model(t)
    prob = torch.softmax(out, 1)

    conf, pred = torch.max(prob, 1)

    return arr, t, conf.item(), pred.item()

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("predict.html")

@app.route("/chat")
def chat():
    return render_template("chat.html")

@app.route("/gradcam")
def gradcam():
    return render_template("gradcam.html")

# ---------------- PREDICT ----------------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    file = request.files.get("file")

    img, _, conf, pred = run(file)

    return jsonify({
        "class": CLASS_NAMES[pred],
        "confidence": conf,
        "image": encode(img)
    })

@app.route("/api/tts", methods=["POST"])
def tts():
    text = request.json.get("text", "")
    chunk = request.json.get("chunk", "")

    target = chunk if chunk else text

    r = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}",
        headers={
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json"
        },
        json={
            "text": target,
            "model_id": "eleven_multilingual_v2"
        }
    )

    if r.status_code != 200:
        return jsonify({"error": r.text}), 500

    return jsonify({
        "audio": base64.b64encode(r.content).decode()
    })
# ---------------- GRADCAM (FIXED) ----------------
@app.route("/api/gradcam", methods=["POST"])
def api_gradcam():
    file = request.files.get("file")

    img, tensor, conf, pred = run(file)

    model.train()
    torch.set_grad_enabled(True)
    model.zero_grad()

    heatmap = cam.generate(tensor, pred)

    model.eval()

    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    return jsonify({
        "class": CLASS_NAMES[pred],
        "confidence": conf,
        "heatmap": encode(overlay)
    })

# ---------------- CHAT (FIXED SDK OBJECT PARSING) ----------------
@app.route("/api/chat", methods=["POST"])
def api_chat():
    msg = request.json.get("message", "")

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {
                    "role": "system",
                    "content": "You are DermVision, a helpful skin analysis assistant."
                },
                {
                    "role": "user",
                    "content": msg
                }
            ]
        )

        text = ""

        if response.output:
            for item in response.output:
                for c in getattr(item, "content", []) or []:
                    text += getattr(c, "text", "") or ""

        return jsonify({"reply": text or "No response"})

    except Exception as e:
        return jsonify({"reply": str(e)})

# ---------------- STREAM CHAT (OPTIONAL FIXED) ----------------
@app.route("/chat_stream", methods=["POST"])
def chat_stream():
    msg = request.json.get("message", "")

    def generate():
        try:
            response = client.responses.create(
                prompt={
                    "id": PROMPT_ID,
                    "version": PROMPT_VERSION
                },
                input=msg,
                stream=True
            )

            for event in response:
                try:
                    for item in event.output:
                        for c in item.content:
                            token = getattr(c, "text", "")
                            if token:
                                yield f"data: {token}\n\n"
                except:
                    continue

            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: ERROR: {str(e)}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")



# ---------------- START ----------------
if __name__ == "__main__":
    load_model()
    warmup()
    app.run(host="0.0.0.0", port=5000, debug=True)
import os, json, io, base64
import torch
import torchvision
from torchvision import transforms
from PIL import Image

def init():
    global model, device, transform, classes
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classes = ["crushed", "torn", "water-damaged", "intact-flagged"]
    model_dir = os.getenv("AZUREML_MODEL_DIR", ".")
    model_path = None
    for root, _, files in os.walk(model_dir):
        for f in files:
            if f == "model.pt":
                model_path = os.path.join(root, f)
    m = torchvision.models.resnet50(weights=None)
    m.fc = torch.nn.Linear(m.fc.in_features, len(classes))
    state = torch.load(model_path, map_location=device)
    m.load_state_dict(state)
    m.eval()
    model = m.to(device)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def run(raw_data):
    data = json.loads(raw_data)
    img_b64 = data["image_base64"]
    img_bytes = base64.b64decode(img_b64)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(torch.argmax(probs))
    return {
        "predicted_class": classes[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": {c: float(p) for c, p in zip(classes, probs)},
    }

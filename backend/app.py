from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)

# Ruta del modelo
MODEL_PATH = "modelo_entrenadomedico3"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Mapeo del índice → clase real (sin clase 2)
id2label = {
    0: 1, 1: 3, 2: 4, 3: 5, 4: 6, 5: 7, 6: 8,
    7: 9, 8: 10, 9: 11, 10: 12, 11: 13, 12: 14,
}

CSV_PATH = "registro_web.csv"

def normalizar_texto(texto):
    texto = texto.upper()
    texto = re.sub(r"[^a-zA-Z0-9áéíóúñÁÉÍÓÚÑ\\s]", " ", texto)
    texto = re.sub(r"\\s+", " ", texto)
    return texto.strip()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    texto_original = data.get("text", "")
    texto = normalizar_texto(texto_original)

    inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1).squeeze()

    topk = torch.topk(probs, k=3)
    resultados = [
        {"clase": f"Clase {id2label[i.item()]}", "prob": round(p.item() * 100, 2)}
        for i, p in zip(topk.indices, topk.values)
    ]

    fila = {
        "texto": texto_original,
        "clase_1": resultados[0]["clase"], "prob_1": resultados[0]["prob"],
        "clase_2": resultados[1]["clase"], "prob_2": resultados[1]["prob"],
        "clase_3": resultados[2]["clase"], "prob_3": resultados[2]["prob"],
        "timestamp": datetime.now().isoformat()
    }

    df = pd.DataFrame([fila])
    if os.path.exists(CSV_PATH):
        df.to_csv(CSV_PATH, mode="a", index=False, header=False)
    else:
        df.to_csv(CSV_PATH, index=False)

    return jsonify({"predicciones": resultados})

@app.route("/historial")
def historial():
    if not os.path.exists(CSV_PATH):
        return jsonify([])
    df = pd.read_csv(CSV_PATH)
    return jsonify(df.to_dict(orient="records"))

@app.route("/descargar")
def descargar():
    if os.path.exists(CSV_PATH):
        return send_file(CSV_PATH, as_attachment=True)
    else:
        return "No hay historial disponible.", 404

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import torch
import os
import re
from io import BytesIO
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(_name_)
CORS(app)

# Modelo en Hugging Face
MODEL_PATH = "martinjuanes/medicoTFGGVA"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Mapeo de clases del modelo a clases reales
id2label = {
    0: 1,
    1: 3,
    2: 4,
    3: 5,
    4: 6,
    5: 7,
    6: 8,
    7: 9,
    8: 10,
    9: 11,
    10: 12,
    11: 13,
    12: 14
}

CSV_PATH = "registro_web.csv"


def normalizar_texto(texto):
    texto = str(texto).lower()
    texto = re.sub(r"[^a-zA-Z0-9áéíóúñÁÉÍÓÚÑ\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()


def clasificar_texto(texto_original):
    texto = normalizar_texto(texto_original)

    inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1).squeeze()

    topk = torch.topk(probs, k=3)
    resultados = [
        {
            "clase": f"Clase {id2label[i.item()]}",
            "prob": round(p.item() * 100, 2)
        }
        for i, p in zip(topk.indices, topk.values)
    ]

    return resultados


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "No se ha proporcionado texto."}), 400

    texto_original = data.get("text", "")
    resultados = clasificar_texto(texto_original)

    fila = {
        "texto": texto_original,
        "clase_1": resultados[0]["clase"],
        "prob_1": resultados[0]["prob"],
        "clase_2": resultados[1]["clase"],
        "prob_2": resultados[1]["prob"],
        "clase_3": resultados[2]["clase"],
        "prob_3": resultados[2]["prob"],
        "timestamp": datetime.now().isoformat()
    }

    df = pd.DataFrame([fila])
    if os.path.exists(CSV_PATH):
        df.to_csv(CSV_PATH, mode="a", index=False, header=False)
    else:
        df.to_csv(CSV_PATH, index=False)

    return jsonify({"predicciones": resultados})


@app.route("/predict_excel", methods=["POST"])
def predict_excel():
    if "file" not in request.files:
        return jsonify({"error": "No se ha enviado ningún archivo."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No se ha seleccionado ningún archivo."}), 400

    try:
        if file.filename.lower().endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
    except Exception:
        return jsonify({
            "error": "No se pudo leer el archivo. Asegúrate de que es un Excel o CSV válido."
        }), 400

    if "texto" not in df.columns:
        return jsonify({
            "error": "El archivo debe contener una columna llamada 'texto'."
        }), 400

    resultados_excel = []

    for texto in df["texto"].fillna(""):
        top3 = clasificar_texto(texto)
        resultados_excel.append({
            "clase_1": top3[0]["clase"],
            "prob_1": top3[0]["prob"],
            "clase_2": top3[1]["clase"],
            "prob_2": top3[1]["prob"],
            "clase_3": top3[2]["clase"],
            "prob_3": top3[2]["prob"]
        })

    df_resultado = pd.concat(
        [df.reset_index(drop=True), pd.DataFrame(resultados_excel)],
        axis=1
    )

    output = BytesIO()
    df_resultado.to_excel(output, index=False, engine="openpyxl")
    output.seek(0)

    return send_file(
        output,
        as_attachment=True,
        download_name="resultados_clasificados.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


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


if _name_ == "_main_":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))

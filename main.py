from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel
from ultralytics import YOLO

RAIZ = Path(__file__).resolve().parent
CAMINHO_PESO = RAIZ / "pesos" / "best.pt"
EXTENSOES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

app = FastAPI(title="Sentinela")
modelo = None


class Caixa(BaseModel):
    x_min: int
    y_min: int
    x_max: int
    y_max: int


class Deteccao(BaseModel):
    class_name: str
    confidence: float
    box: Caixa


class Resposta(BaseModel):
    status: str
    detections: list[Deteccao]

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "success",
                "detections": [
                    {
                        "class_name": "lion",
                        "confidence": 0.92,
                        "box": {
                            "x_min": 150,
                            "y_min": 200,
                            "x_max": 250,
                            "y_max": 300,
                        },
                    }
                ],
            }
        }
    }


def carregar_modelo():
    global modelo
    if modelo is None:
        if not CAMINHO_PESO.is_file():
            raise HTTPException(
                status_code=500,
                detail="peso do modelo não encontrado em pesos/best.pt",
            )
        modelo = YOLO(str(CAMINHO_PESO))
    return modelo


@app.post("/detect", response_model=Resposta)
async def detectar(file: UploadFile = File(...)):
    nome = Path(file.filename or "").name
    sufixo = Path(nome).suffix.lower()
    tipo = (file.content_type or "").split(";")[0].lower()
    tipos_aceitos = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
    if sufixo not in EXTENSOES and tipo not in tipos_aceitos:
        raise HTTPException(status_code=400, detail="envie uma imagem jpg, png, webp ou bmp")

    conteudo = await file.read()
    if not conteudo:
        raise HTTPException(status_code=400, detail="arquivo de imagem vazio")

    try:
        imagem = Image.open(BytesIO(conteudo)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="arquivo não é uma imagem válida")

    detector = carregar_modelo()
    resultado = detector.predict(
        source=imagem,
        imgsz=640,
        conf=0.25,
        device="cpu",
        verbose=False,
    )[0]

    deteccoes = []
    if resultado.boxes is not None:
        for caixa in resultado.boxes:
            x_min, y_min, x_max, y_max = caixa.xyxy[0].tolist()
            deteccoes.append(
                {
                    "class_name": detector.names[int(caixa.cls)],
                    "confidence": round(float(caixa.conf), 4),
                    "box": {
                        "x_min": int(round(x_min)),
                        "y_min": int(round(y_min)),
                        "x_max": int(round(x_max)),
                        "y_max": int(round(y_max)),
                    },
                }
            )
    deteccoes.sort(key=lambda item: item["confidence"], reverse=True)
    return {"status": "success", "detections": deteccoes}

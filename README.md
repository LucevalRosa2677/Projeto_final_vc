# Sentinela

Software de detecção de animais. O autor é Luceval Rosa.

O Sentinela recebe uma foto e devolve um JSON com os animais encontrados, a confiança e a caixa de cada um. A resposta não traz a imagem desenhada.

## Dataset

O dataset é o [Labeled animal dataset for YOLO training](https://www.kaggle.com/datasets/nhutrinhanna/labeled-animal-dataset-for-yolo-training), de Nhu Trinh, licença Apache 2.0.

São 18 classes. Os nomes abaixo são os que a API devolve:

| Código | Classe |
| --- | --- |
| 0 | antelope |
| 1 | bat |
| 2 | bear |
| 3 | bee |
| 4 | bison |
| 5 | boar |
| 6 | cheetah |
| 7 | coyote |
| 8 | fox |
| 9 | hyena |
| 10 | leopard |
| 11 | lion |
| 12 | ox |
| 13 | porcupine |
| 14 | rhinoceros |
| 15 | snake |
| 16 | tiger |
| 17 | wolf |

O desafio é localizar esses animais na foto e informar a posição de cada um, sem devolver a imagem processada.

As imagens usadas no treino não entram no repositório. O arquivo `data.yaml` descreve a divisão em treino, validação e teste.

## Modelo

O modelo é o YOLO11n, da Ultralytics. Foi escolhido por ser a versão leve da família YOLO11: cabe na API e roda em um computador sem placa NVIDIA. O treino foi feito em GPU no Google Colab.

Na validação, o mAP50 ficou em 0,8349 e o mAP50-95 em 0,6671.

O peso treinado está em `pesos/best.pt`.

## Como executar

Na pasta do projeto:

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Se o comando `uvicorn` não for encontrado:

```bash
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload
```

Abra http://127.0.0.1:8000/docs. No `POST /detect`, envie uma imagem pelo campo `file`.

## Exemplo de JSON

```json
{
  "status": "success",
  "detections": [
    {
      "class_name": "wolf",
      "confidence": 0.8913,
      "box": {
        "x_min": 2,
        "y_min": 1,
        "x_max": 338,
        "y_max": 475
      }
    },
    {
      "class_name": "wolf",
      "confidence": 0.858,
      "box": {
        "x_min": 283,
        "y_min": 189,
        "x_max": 457,
        "y_max": 463
      }
    }
  ]
}
```

`class_name` é a classe. `confidence` é a confiança, de 0 a 1. `box` traz os cantos da caixa em pixels: `x_min`, `y_min`, `x_max` e `y_max`.

## Quadro Kanban

https://trello.com/b/LIi74H9o/sentinela

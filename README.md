#### Hadhari Arabic spam detection model.

https://huggingface.co/spaces/mabosaimi/hadhari

---
**Prediction API Endpoint:** https://mabosaimi-hadhari.hf.space/predict

Expects a **POST** request with the JSON payload:
```json
{ "text": "الرسالة" }
```

**Response:**
```json
{
  "prediction": 0,
  "confidence": 0.96
}
```
**Prediction:** 1 = spam, 0 = not spam.

**Confidence:** model's confidence percentage in the prediction.

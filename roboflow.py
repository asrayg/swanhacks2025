import cv2
import requests
import base64
import json

API_KEY = "rf_PKhpEEZp8sXuqAKPGMJWJAL0Y6i2"
MODEL = "swanhacks-v2-lb331-instant-2"
VERSION = 1

MODEL_URL = "https://detect.roboflow.com/swanhacks-v2-lb331-instant-2/1"

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Encode frame as JPEG
    _, buffer = cv2.imencode(".jpg", frame)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    # Send to Roboflow
    resp = requests.post(
        MODEL_URL,
        params={"api_key": API_KEY, "format": "json"},
        data=img_base64,
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )

    preds = resp.json()
    print(preds)   # ← debug output

    # Draw predictions
    if "predictions" in preds:
        for p in preds["predictions"]:
            x, y = int(p["x"]), int(p["y"])
            w, h = int(p["width"]), int(p["height"])
            cls = p["class"]
            conf = p["confidence"]

            x1 = x - w // 2
            y1 = y - h // 2
            x2 = x + w // 2
            y2 = y + h // 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{cls} {conf:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    cv2.imshow("Live Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

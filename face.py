import cv2
import base64
import requests
import time

# ----------------------------
# CONFIG
# ----------------------------

ROBOFLOW_MODEL = "swanhacks-v2-lb331"
ROBOFLOW_VERSION = "3"
API_KEY = "rf_PKhpEEZp8sXuqAKPGMJWJAL0Y6i2"

INFER_URL = f"https://detect.roboflow.com/{ROBOFLOW_MODEL}/{ROBOFLOW_VERSION}?api_key={API_KEY}"

# ----------------------------
# HELPERS
# ----------------------------

def infer_image(frame):
    """
    Sends a single webcam frame to Roboflow and returns detections.
    """

    # Encode frame as JPG
    _, buffer = cv2.imencode(".jpg", frame)
    img_bytes = buffer.tobytes()

    # Base64 encode
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    # Send request to Roboflow
    resp = requests.post(
        INFER_URL,
        data=img_b64,
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )

    return resp.json()

# ----------------------------
# MAIN LOOP
# ----------------------------

def main():
    print("🔥 Starting webcam…")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ ERROR: Could not open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Run inference
        result = infer_image(frame)

        # Draw detections
        if "predictions" in result:
            for pred in result["predictions"]:
                x = int(pred["x"])
                y = int(pred["y"])
                w = int(pred["width"])
                h = int(pred["height"])
                cls = pred["class"]
                conf = pred["confidence"]

                # Bounding box coordinates
                x1 = x - w // 2
                y1 = y - h // 2
                x2 = x + w // 2
                y2 = y + h // 2

                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(
                    frame,
                    f"{cls} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,0),
                    2
                )

        # Display
        cv2.imshow("Roboflow Facial Detection (Live)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
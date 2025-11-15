import cv2
import base64
import time
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def frame_to_base64(frame):
    _, jpeg = cv2.imencode(".jpg", frame)
    return base64.b64encode(jpeg.tobytes()).decode("utf-8")

def analyze_frame(frame):
    img_b64 = frame_to_base64(frame)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe what is happening in this webcam frame."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}"
                        }
                    }
                ]
            }
        ]
    )

    return response.choices[0].message.content

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return

    print("🎥 Webcam started. Press Ctrl+C to stop.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            start = time.time()
            desc = analyze_frame(frame)
            duration = time.time() - start

            print(f"GPT ({duration:.2f}s): {desc}")

            cv2.imshow("Live Webcam (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        pass

    print("🛑 Stopped.")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

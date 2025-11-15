import cv2
import base64
import time
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def frame_to_base64(frame):
    frame_small = cv2.resize(frame, (320, 180))  # super lightweight

    # Compress heavily to reduce image size
    _, jpeg = cv2.imencode(".jpg", frame_small, [cv2.IMWRITE_JPEG_QUALITY, 30])

    return base64.b64encode(jpeg.tobytes()).decode("utf-8")


def analyze_frame(frame):
    img_b64 = frame_to_base64(frame)

    response = client.chat.completions.create(
        model="gpt-4o-mini",      # smaller, cheaper, faster
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this scene briefly."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
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

    print("🎥 Webcam started. Press q to quit.\n")

    last_time = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            now = time.time()

            if now - last_time > 1.2:
                start = time.time()
                desc = analyze_frame(frame)
                duration = time.time() - start

                print(f"\n🤖 GPT ({duration:.2f}s): {desc}")
                last_time = now

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

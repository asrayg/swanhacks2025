import cv2
import base64
import time
import json
import tkinter as tk
from threading import Thread

from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

context_history = []  # store last 10 descriptions


def show_alert(message):
    def _popup():
        root = tk.Tk()
        root.title("⚠️ ALERT DETECTED")
        root.geometry("400x200")
        label = tk.Label(root, text=message, font=("Arial", 14), wraplength=350)
        label.pack(pady=20)
        button = tk.Button(root, text="OK", command=root.destroy)
        button.pack()
        root.mainloop()

    Thread(target=_popup).start()


def frame_to_base64(frame):
    frame_small = cv2.resize(frame, (320, 180))
    _, jpeg = cv2.imencode(".jpg", frame_small, [cv2.IMWRITE_JPEG_QUALITY, 30])
    return base64.b64encode(jpeg.tobytes()).decode("utf-8")


def analyze_frame(frame):
    img_b64 = frame_to_base64(frame)

    history_text = "\n".join(context_history[-5:]) or "None yet."

    system_prompt = (
    "You are a realtime safety monitoring AI.\n"
    "You receive webcam frames and short descriptions of previous frames.\n\n"
    f"Previous context:\n{history_text}\n\n"
    "TASKS:\n"
    "1. Detect aggression:\n"
    "   - clenched fists\n"
    "   - raised voice indicators\n"
    "   - rapid movement toward someone\n"
    "   - angry facial expression\n"
    "   - hitting / throwing objects\n\n"
    "2. Detect medical anomalies:\n"
    "   - allergic reactions\n"
    "   - brown urine container present\n"
    "   - sweating, pale face\n"
    "   - shaking, fainting\n"
    "   - peanuts or allergens near subject\n"
    "   - breathing issues\n"
    "   - visible blood or injury\n\n"
    "Return STRICT JSON ONLY:\n"
    "{\n"
    "  \"description\": \"short scene description\",\n"
    "  \"aggression\": true/false,\n"
    "  \"aggression_level\": 0-10,\n"
    "  \"medical\": true/false,\n"
    "  \"medical_issue\": \"string or null\"\n"
    "}\n"
    )


    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze the frame."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                    }
                ]
            }
        ]
    )

    # parse JSON
    result = json.loads(response.choices[0].message.content)
    return result


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Webcam not found.")
        return

    print("\n🎥 Monitoring started. Press q to quit.\n")

    last_analyze = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            now = time.time()

            if now - last_analyze > 1.5:
                result = analyze_frame(frame)

                # update context
                context_history.append(result["description"])
                if len(context_history) > 10:
                    context_history.pop(0)

                print(f"🤖 {result}")

                # trigger alert
                if result["aggression"]:
                    show_alert(f"AGGRESSION DETECTED (Level {result['aggression_level']})")

                if result["medical"]:
                    show_alert(f"MEDICAL ISSUE DETECTED: {result['medical_issue']}")

                last_analyze = now

            # show webcam preview
            cv2.imshow("Live Monitor (press q to exit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        pass

    print("🛑 Monitoring stopped.")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

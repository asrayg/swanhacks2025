import cv2
import base64
import time
import json
from openai import OpenAI
from dotenv import load_dotenv
import os
from datetime import datetime
import pyaudio
import wave
import threading
import re

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------
# AUDIO RECORDER
# -----------------------------
AUDIO_FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024


class AudioRecorder:
    def __init__(self, filename):
        self.filename = filename
        self.frames = []
        self.is_recording = False
        self.audio = pyaudio.PyAudio()
        
    def start_recording(self):
        self.is_recording = True
        self.stream = self.audio.open(
            format=AUDIO_FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        
        def record():
            while self.is_recording:
                data = self.stream.read(CHUNK, exception_on_overflow=False)
                self.frames.append(data)
        
        self.thread = threading.Thread(target=record)
        self.thread.start()
        print("🎤 Audio recording started")
    
    def stop_recording(self):
        self.is_recording = False
        self.thread.join()
        self.stream.stop_stream()
        self.stream.close()
        
        wf = wave.open(self.filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.audio.get_sample_size(AUDIO_FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        self.audio.terminate()
        print(f"🎤 Audio saved to: {self.filename}")


def extract_json(text):
    """
    Extract JSON object from a GPT response that may contain noise.
    """
    if not text:
        return None
    
    # Find first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    
    json_str = match.group(0)
    
    try:
        return json.loads(json_str)
    except:
        return None


# -----------------------------
# FRAME → BASE64
# -----------------------------
def frame_to_base64(frame):
    frame_small = cv2.resize(frame, (320, 180))
    _, jpeg = cv2.imencode(".jpg", frame_small, [cv2.IMWRITE_JPEG_QUALITY, 30])
    return base64.b64encode(jpeg.tobytes()).decode("utf-8")


# -----------------------------
# GPT VISION + SAFETY ANALYSIS
# -----------------------------
context_history = []

def analyze_frame(frame):
    img_b64 = frame_to_base64(frame)

    context_text = "\n".join(context_history[-5:]) or "None yet"

    system_prompt = (
        "You are a realtime safety monitoring AI.\n\n"
        f"Previous context:\n{context_text}\n\n"
        "TASKS:\n"
        "1. Detect aggression:\n"
        "   - clenched fists\n"
        "   - raised voice/open-mouth yelling posture\n"
        "   - rapid movement toward someone\n"
        "   - angry facial expression\n"
        "   - throwing or hitting objects\n\n"
        "2. Detect medical anomalies:\n"
        "   - allergic reactions\n"
        "   - peanuts or allergens near patient\n"
        "   - brown urine container\n"
        "   - pale face, sweating\n"
        "   - fainting signs\n"
        "   - blood, visible injury\n\n"
        "3. Provide routing instructions:\n"
        "   - If aggression: notify security\n"
        "   - If allergic hazard: page allergy specialist\n"
        "   - If injury: notify medical response\n"
        "   - If normal: no routing needed\n\n"
        "Return STRICT JSON ONLY:\n"
        "{\n"
        "  \"description\": \"short scene summary\",\n"
        "  \"aggression\": true/false,\n"
        "  \"aggression_level\": 0-10,\n"
        "  \"medical\": true/false,\n"
        "  \"medical_issue\": \"string or null\",\n"
        "  \"routing\": \"security | doctor | none | allergy | injury | emergency\"\n"
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
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                    }
                ]
            }
        ]
    )

    raw = response.choices[0].message.content
    result = extract_json(raw)

    if result is None:
        print("⚠️ GPT returned non-JSON:", raw)
        return {
            "description": "JSON parse error",
            "aggression": False,
            "aggression_level": 0,
            "medical": False,
            "medical_issue": None,
            "routing": "none"
        }
    return result


# -----------------------------
# AUDIO → TEXT
# -----------------------------
def transcribe_audio(audio_filename):
    try:
        with open(audio_filename, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        return transcript.text
    except Exception as e:
        print(f"⚠️ Audio transcription failed: {e}")
        return None


# -----------------------------
# FINAL REPORT
# -----------------------------
def generate_report(events, audio_transcript):
    history_json = json.dumps(events, indent=2)

    prompt = f"""
You are a medical & security incident analyst.

Session visual events (JSON list):
{history_json}

Audio transcript:
{audio_transcript}

Generate a structured incident summary including:
- Overall scene summary
- Aggression evaluation
- Medical concerns
- Timeline of observations
- Recommended actions
- Notes for security or doctor routing
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You analyze safety, aggression, and medical incidents."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


# -----------------------------
# SAVE REPORT TO FILE
# -----------------------------
def save_report(report, audio_file):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"session_report_{ts}.txt"

    with open(filename, "w") as f:
        f.write("=== SESSION REPORT ===\n\n")
        f.write(report)
        f.write("\n\nAudio file: " + audio_file)

    print(f"\n📄 Report saved to: {filename}")
    return filename


# -----------------------------
# MAIN LOOP
# -----------------------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam error")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_filename = f"session_audio_{timestamp}.wav"
    audio_recorder = AudioRecorder(audio_filename)

    print("🎥 Starting session...")
    print("🎤 Audio recording starting...\n")
    
    audio_recorder.start_recording()

    last_frame_time = 0
    session_start = time.time()

    events = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            now = time.time()

            # Throttle GPT calls
            if now - last_frame_time > 1.5:
                result = analyze_frame(frame)
                context_history.append(result["description"])
                if len(context_history) > 10:
                    context_history.pop(0)

                elapsed = now - session_start
                result["timestamp"] = elapsed
                events.append(result)

                print("\n----------------------")
                print(f"Time: {elapsed:.1f}s")
                print(f"Scene: {result['description']}")
                print(f"Aggression: {result['aggression']} (Level {result['aggression_level']})")
                print(f"Medical: {result['medical']} - {result['medical_issue']}")
                print(f"Routing: {result['routing']}")
                print("----------------------")

                last_frame_time = now

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        pass

    print("\n🛑 Ending session...")
    audio_recorder.stop_recording()

    print("\n🎤 Transcribing audio...")
    audio_transcript = transcribe_audio(audio_filename)

    print("\n📄 Generating final session report...")
    report = generate_report(events, audio_transcript)

    save_report(report, audio_filename)

    cap.release()
    cv2.destroyAllWindows()

    print("\n✅ Session complete.")


if __name__ == "__main__":
    main()

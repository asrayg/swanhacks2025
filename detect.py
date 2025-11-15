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
# AUDIO RECORDER (continuous)
# -----------------------------
AUDIO_FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024


class AudioRecorder:
    def __init__(self):
        self.frames = []
        self.is_recording = False
        self.audio = pyaudio.PyAudio()

    def start(self):
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
        print("🎤 Audio recording started (continuous)")

    def stop_and_save_full_audio(self, filename):
        """Save entire session audio."""
        self.is_recording = False
        self.thread.join()
        self.stream.stop_stream()
        self.stream.close()

        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.audio.get_sample_size(AUDIO_FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()

        self.audio.terminate()
        print(f"🎤 Full session audio saved to: {filename}")

    def save_chunk(self, chunk_filename):
        """Save last few seconds for real-time transcription."""
        # Copy last 5 seconds of audio
        num_frames = int(5 * RATE / CHUNK)
        chunk_data = self.frames[-num_frames:] if len(self.frames) >= num_frames else self.frames

        wf = wave.open(chunk_filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.audio.get_sample_size(AUDIO_FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(chunk_data))
        wf.close()
        return True


# -----------------------------
# SAFE JSON EXTRACTOR
# -----------------------------
def extract_json(text):
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
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
# REALTIME CONTEXT MEMORY
# -----------------------------
visual_context = []
audio_context = []


def detect_audio_keywords(text):
    if not text:
        return None
    
    text = text.lower()

    keywords = {
        "it hurts": "reported pain",
        "hurt": "reported pain",
        "help": "call for help",
        "stop": "possible distress",
        "please stop": "possible assault",
        "i can't breathe": "respiratory distress",
        "allergic": "possible allergic reaction",
        "peanut": "allergy hazard",
        "chest": "chest pain",
        "pain": "pain reported",
        "faint": "fainting risk",
    }

    for key, issue in keywords.items():
        if key in text:
            return issue

    return None

# -----------------------------
# GPT VISION + SAFETY ANALYSIS
# -----------------------------
def analyze_frame(frame):
    img_b64 = frame_to_base64(frame)

    visual_text = "\n".join(visual_context[-5:]) or "None yet"
    audio_text = "\n".join(audio_context[-3:]) or "No audio context"

    system_prompt = (
        "You are a realtime multimodal safety monitoring AI.\n\n"
        f"Recent Visual Context:\n{visual_text}\n\n"
        f"Recent Audio Context:\n{audio_text}\n\n"
        "TASKS:\n"
        "1. Detect aggression:\n"
        "   - yelling or angry tone (audio)\n"
        "   - shouting ('stop', 'no', 'help')\n"
        "   - open-mouth yelling posture\n"
        "   - clenched fists\n"
        "   - moving toward someone rapidly\n"
        "   - smashing or throwing objects\n\n"
        "2. Detect medical anomalies:\n"
        "   - phrases like 'I can't breathe', 'it hurts', 'help me'\n"
        "   - allergic reaction related language\n"
        "   - peanuts or allergens visible \n"
        "   - sweating, pale face, fainting signs\n"
        "   - brown urine container\n"
        "   - blood or injury\n\n"
        "3. Provide routing:\n"
        "   security | doctor | allergy | injury | emergency | none\n\n"
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
                    {"type": "text", "text": "Analyze the frame with context."},
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
            "description": "json_error",
            "aggression": False,
            "aggression_level": 0,
            "medical": False,
            "medical_issue": None,
            "routing": "none"
        }

    # HARD OVERRIDE FROM AUDIO CONTEXT
    if audio_context:
        last_audio = " ".join(audio_context[-3:]).lower()
        issue = detect_audio_keywords(last_audio)

        if issue:
            result["medical"] = True
            result["medical_issue"] = issue
            result["routing"] = "emergency" if "breathe" in last_audio else "doctor"

    return result


# -----------------------------
# AUDIO → TEXT (Realtime Chunk)
# -----------------------------
def transcribe_chunk(chunk_file):
    try:
        with open(chunk_file, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        return transcript.text
    except:
        return None


# -----------------------------
# FINAL REPORT
# -----------------------------
def generate_report(events, full_audio_transcript):
    history_json = json.dumps(events, indent=2)

    prompt = f"""
Full Event JSON:
{history_json}

Full Audio Transcript:
{full_audio_transcript}

Write a structured incident report with:
- Summary
- Aggression analysis
- Medical findings
- Timeline
- Recommended actions
- Routing notes
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You summarize multimodal safety sessions."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


# -----------------------------
# SAVE REPORT
# -----------------------------
def save_report(report, audio_file):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"session_report_{ts}.txt"

    with open(filename, "w") as f:
        f.write("=== SESSION REPORT ===\n\n")
        f.write(report)
        f.write("\n\nAudio File: " + audio_file)

    print(f"📄 Report saved: {filename}")
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
    full_audio_filename = f"session_audio_{timestamp}.wav"

    # Start continuous audio recording
    audio_rec = AudioRecorder()
    audio_rec.start()

    print("🎥 Session started (press q to quit)")
    print("🎤 Audio + video monitoring active\n")

    last_frame_time = 0
    last_audio_time = 0
    session_start = time.time()

    events = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            now = time.time()

            # ------------------------------
            # REALTIME AUDIO CHUNK EVERY 5s
            # ------------------------------
            if now - last_audio_time > 5:
                chunk_file = "temp_audio_chunk.wav"
                audio_rec.save_chunk(chunk_file)

                text = transcribe_chunk(chunk_file)
                issue = detect_audio_keywords(text)
                if issue:
                    print(f"🚨 AUDIO FLAG DETECTED: {issue}")
                    audio_context.append(f"ALERT: {issue}")
                    if len(audio_context) > 5:
                        audio_context.pop(0)
                    print(f"🎧 Audio snippet: {text}")

                last_audio_time = now

            # ------------------------------
            # GPT FRAME ANALYSIS
            # ------------------------------
            if now - last_frame_time > 1.5:
                result = analyze_frame(frame)

                visual_context.append(result["description"])
                if len(visual_context) > 10:
                    visual_context.pop(0)

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

    # Save full session audio
    audio_rec.stop_and_save_full_audio(full_audio_filename)

    print("\n🎤 Transcribing full audio...")
    full_transcript = transcribe_chunk(full_audio_filename)

    print("\n📄 Generating final session report...")
    report = generate_report(events, full_transcript)

    save_report(report, full_audio_filename)

    cap.release()
    cv2.destroyAllWindows()

    print("\n✅ Session complete.")


if __name__ == "__main__":
    main()

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
import random
import subprocess
from supabase import create_client, Client
from PIL import Image, ImageDraw, ImageFont
from driver import OLED_1in51, OLED_WIDTH, OLED_HEIGHT
import numpy as np
oled = None

# OLED control lock - set by main.py to coordinate screen access
# When JARVIS or translator have the lock, detect should pause updates
oled_control_lock = None

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))



# -----------------------------
# SUPABASE CONFIGURATION
# -----------------------------
SUPABASE_URL = "https://lfvpbnzpsxxgppnlzhln.supabase.co"
SUPABASE_SERVICE_ROLE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImxmdnBibnpwc3h4Z3Bwbmx6aGxuIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MzIzMzg5MCwiZXhwIjoyMDc4ODA5ODkwfQ.W8W03HO2-uRUrwBqfDFnd203s1NMUkgYiI3oRlPqHBg"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# -----------------------------
# CONFIGURABLE CONTEXT
# -----------------------------
vital_signs = {
    "heart_rate": "85 bpm",
    "oxygen_level": "95%",
    "height": "5'10\""
}

doctors_available = {
    "allergy": ("Dr. Patel", 3),
    "injury": ("Dr. Wong", 5),
    "emergency": ("Dr. Evans", 1),
    "general": ("Dr. Lee", 4)
}

security_units = {
    "Unit A": "2 minutes away",
    "Unit B": "4 minutes away"
}

def oled_print(text, size=9):
    global oled, oled_control_lock
    
    # Check if JARVIS/translator has control - if so, skip update
    if oled_control_lock is not None:
        # Try to acquire the lock with timeout=0 (non-blocking)
        acquired = oled_control_lock.acquire(blocking=False)
        if not acquired:
            # Lock is held by JARVIS or translator - skip this update
            return
        else:
            # We got the lock, release it immediately and proceed
            oled_control_lock.release()

    # Create blank image
    image = Image.new("1", (OLED_WIDTH, OLED_HEIGHT), 255)
    draw = ImageDraw.Draw(image)

    # Smaller readable font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except:
        font = ImageFont.load_default()

    # Split text into lines (approx 18 chars/line)
    words = text.split()
    line = ""
    lines = []
    for w in words:
        if len(line) + len(w) < 24:
            line += " " + w
        else:
            lines.append(line.strip())
            line = w
    lines.append(line.strip())

    # Draw max 4 lines
    y = 0
    for ln in lines[:4]:
        draw.text((0, y), ln, font=font, fill=0)
        y += size + 3

    # Rotate entire screen upside down
    image = image.rotate(180)

    # Send output
    buf = oled.getbuffer(image)
    oled.ShowImage(buf)

    # Slow output so text doesn't flicker/flash too fast
    time.sleep(0.4)


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
        self.audio = None
        self.stream = None

    def start(self):
        print("🎤 Initializing audio...")
        try:
            import pyaudio
            self.audio = pyaudio.PyAudio()
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=44100,
                input=True,
                frames_per_buffer=1024
            )
        except Exception as e:
            print("❌ Mic initialization failed:", e)
            print("➡️ Continuing WITHOUT audio.")
            self.audio = None
            return  # disable audio completely

        self.is_recording = True

        def record():
            while self.is_recording:
                data = self.stream.read(1024, exception_on_overflow=False)
                self.frames.append(data)

        self.thread = threading.Thread(target=record)
        self.thread.start()
        print("🎤 Audio recording started!")

    def stop_and_save_full_audio(self, filename):
        if not self.audio:
            print("⚠️ No audio was recorded.")
            return

        self.is_recording = False
        self.thread.join()
        self.stream.stop_stream()
        self.stream.close()

        wf = wave.open(filename, "wb")
        wf.setnchannels(1)
        wf.setsampwidth(self.audio.get_sample_size(AUDIO_FORMAT))
        wf.setframerate(44100)
        wf.writeframes(b''.join(self.frames))
        wf.close()

        self.audio.terminate()

    def save_chunk(self, chunk_filename):
        if not self.audio:
            return False

        num_frames = int(5 * RATE / CHUNK)
        chunk_data = (
            self.frames[-num_frames:] if len(self.frames) >= num_frames else self.frames
        )

        wf = wave.open(chunk_filename, "wb")
        wf.setnchannels(1)
        wf.setsampwidth(self.audio.get_sample_size(AUDIO_FORMAT))
        wf.setframerate(44100)
        wf.writeframes(b"".join(chunk_data))
        wf.close()

        return True

    
def capture_frame(path="/dev/shm/frame.jpg"):
    """Reliable single-frame capture using rpicam-still (headless-safe)."""
    cmd = [
        "rpicam-still",
        "-t", "1",        # tiny timeout for capture
        "--width", "640",
        "--height", "480",
        "-n",             # no preview
        "-o", path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return cv2.imread(path)

def detect_shutdown_command(text):
    if not text:
        return False

    text = text.lower()

    shutdown_phrases = [
        "stop monitoring",
        "stop the system",
        "stop detection",
        "shutdown",
        "shut down",
        "end session",
        "terminate session"
    ]

    return any(phrase in text for phrase in shutdown_phrases)

def generate_dynamic_vitals():
    """Generate slightly varying but realistic medical vitals."""
    
    hr = random.randint(75, 95)  # Heart rate bpm
    sys = random.randint(110, 135)  # Systolic BP
    dia = random.randint(70, 90)   # Diastolic BP
    oxy = random.randint(94, 100)  # %
    resp = random.randint(12, 20)  # breaths per minute
    temp = round(random.uniform(97.5, 99.2), 1)  # °F

    return {
        "heart_rate": f"{hr} bpm",
        "blood_pressure": f"{sys}/{dia} mmHg",
        "oxygen": f"{oxy}%",
        "respiration": f"{resp} breaths/min",
        "temperature": f"{temp}°F"
    }


def realtime_routing_alert(result):
    routing = result.get("routing", "none")
    issue = result.get("medical_issue", None)
    aggro = result.get("aggression", False)
    level = result.get("aggression_level", 0)

    print("\n📡 ROUTING STATUS UPDATE ---------------------------")

    # ---------------------------------------------------------
    # 🚨 SECURITY RESPONSE
    # ---------------------------------------------------------
    if routing == "security":
        oled_print("SECURITY DISPATCHED:")
        for unit, eta in security_units.items():
            oled_print(f"    {unit}  ETA {eta}")
        print("--------------------------------------------------")
        return

    if routing == "emergency":
        oled_print("EMERGENCY RESPONSE ACTIVATED:\nNotifying all security units:")
        for unit, eta in security_units.items():
            oled_print(f"       - {unit}  ETA {eta}")
        name, eta = doctors_available["emergency"]
        oled_print(f"    Paging ER Doctor: {name}  ETA {eta} minutes")
        print("--------------------------------------------------")
        return

    # ---------------------------------------------------------
    # 🩺 MEDICAL ROUTES
    # ---------------------------------------------------------
    if routing == "doctor":
        name, eta = doctors_available["general"]
        oled_print(f"DOCTOR PAGED:\n{name}  ETA {eta} minutes\nIssue: {issue}")
        print("--------------------------------------------------")
        return

    if routing == "allergy":
        name, eta = doctors_available["allergy"]
        oled_print(f"ALLERGY SPECIALIST PAGED:\n{name}  ETA {eta} minutes\nTrigger: {issue}")
        print("--------------------------------------------------")
        return

    if routing == "injury":
        name, eta = doctors_available["injury"]
        oled_print(f"TRAUMA/INJURY PHYSICIAN PAGED:\n{name}  ETA {eta} minutes\nIssue: {issue}")
        print("--------------------------------------------------")
        return

    # ---------------------------------------------------------
    # NOTHING HAPPENING → STILL PRINT USEFUL STATUS
    # ---------------------------------------------------------
    if routing == "none":
        dynamic = generate_dynamic_vitals()

        print("✅ No routing required at this moment.")
        oled_print(f"Heart Rate: {dynamic['heart_rate']}\nBlood Pressure: {dynamic['blood_pressure']}\nOxygen Level: {dynamic['oxygen']}\nRespiration: {dynamic['respiration']}\nTemperature: {dynamic['temperature']}")
        print("--------------------------------------------------")
        return

# -----------------------------
# JSON SAFE EXTRACTOR
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
# FRAME BASE64 ENCODE
# -----------------------------
def frame_to_base64(frame):
    frame_small = cv2.resize(frame, (320, 180))
    _, jpeg = cv2.imencode(".jpg", frame_small, [cv2.IMWRITE_JPEG_QUALITY, 30])
    return base64.b64encode(jpeg.tobytes()).decode("utf-8")


# -----------------------------
# CONTEXT MEMORY
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
# GPT FRAME ANALYSIS
# -----------------------------
def analyze_frame(frame):
    img_b64 = frame_to_base64(frame)

    visual_text = "\n".join(visual_context[-5:]) or "None yet"
    audio_text = "\n".join(audio_context[-3:]) or "No audio context"

    system_prompt = f"""
You are a realtime multimodal safety monitoring AI.

VITAL SIGNS:
{vital_signs}

DOCTORS AVAILABLE:
{doctors_available}

SECURITY UNITS:
{security_units}

Recent Visual Context:
{visual_text}

Recent Audio Context:
{audio_text}

TASKS:
1. Detect aggression:
   - yelling, shouting, 'stop', 'help'
   - clenched fists, rapid movement, hitting objects

2. Detect medical anomalies:
   - “I can’t breathe”, “it hurts”, “help me”
   - allergies, peanuts in view
   - fainting, sweating, pale face
   - blood or injury

3. Provide routing:
   security | doctor | allergy | injury | emergency | none

STRICT JSON ONLY:
{{
  "description": "summary",
  "aggression": true/false,
  "aggression_level": 0-10,
  "medical": true/false,
  "medical_issue": "string or null",
  "routing": "security | doctor | allergy | injury | emergency | none"
}}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": "Analyze frame."},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{img_b64}"
                }}
            ]}
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

    # Apply hard audio override
    # HARD OVERRIDE FROM AUDIO CONTEXT
    if audio_context:
        last_audio = " ".join(audio_context[-3:]).lower()
        issue = detect_audio_keywords(last_audio)
        if issue:
            result["medical"] = True
            result["medical_issue"] = issue
            result["routing"] = "emergency" if "breathe" in last_audio else "doctor"

    # -----------------------------
    # 🔥 NEW: HARD RULE ROUTING OVERRIDES
    # -----------------------------

    # If aggression is detected but routing is missing → force security.
    if result["aggression"] and result["routing"] == "none":
        if result["aggression_level"] >= 7:
            result["routing"] = "emergency"
        else:
            result["routing"] = "security"

    # If medical emergency always escalate properly
    if result["medical"]:
        if "breathe" in str(result["medical_issue"]).lower():
            result["routing"] = "emergency"
        elif "allergy" in str(result["medical_issue"]).lower():
            result["routing"] = "allergy"
        else:
            result["routing"] = "doctor"

    return result


# -----------------------------
# AUDIO TRANSCRIPTION
# -----------------------------
def transcribe_chunk(audio_file):
    try:
        with open(audio_file, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f
            )
        return transcript.text
    except:
        return None


# -----------------------------
# FINAL REPORT GENERATION
# -----------------------------
def generate_report(events, audio_transcript):
    history = json.dumps(events, indent=2)

    prompt = f"""
VITAL SIGNS:
{vital_signs}

DOCTORS:
{doctors_available}

SECURITY UNITS:
{security_units}

EVENT JSON (visual + audio):
{history}

AUDIO TRANSCRIPT:
{audio_transcript}

Write a full incident report including:
- Summary
- Patient condition
- Aggression analysis
- Medical findings
- Routing decisions and who is called
- Timeline
- Recommended actions
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Write medical/security incident reports."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def post_to_supabase(report_path, audio_path):
    """Post the report and audio file to Supabase Expo table."""
    try:
        # Read the report text content
        with open(report_path, "r", encoding="utf-8") as f:
            report_content = f.read()
        
        # Get absolute paths for the files
        abs_report_path = os.path.abspath(report_path)
        abs_audio_path = os.path.abspath(audio_path)
        
        # For Video column, store the audio file path
        # You can modify this to upload to Supabase Storage and get a public URL instead
        video_value = abs_audio_path
        
        # Generate a unique ID (or let Supabase auto-generate)
        # We'll let Supabase auto-generate the ID by not providing it
        
        print("📤 Posting to Supabase...")
        
        # Insert into Expo table
        response = supabase.table("Expo").insert({
            "Report": report_content,
            "Video": video_value
        }).execute()
        
        if response.data:
            print(f"✅ Successfully posted to Supabase!")
            print(f"   Record ID: {response.data[0].get('id', 'N/A')}")
            print(f"   Report length: {len(report_content)} characters")
            print(f"   Audio file: {abs_audio_path}")
        else:
            print("⚠️ No data returned from Supabase insert")
            
    except Exception as e:
        print(f"❌ Error posting to Supabase: {e}")
        raise


# -----------------------------
# SAVE REPORT + MEDIA
# -----------------------------
def save_output(report, audio_file, frames, session_folder):
    report_path = os.path.join(session_folder, "report.txt")
    audio_path = os.path.join(session_folder, audio_file)

    # Save report
    with open(report_path, "w") as f:
        f.write(report)

    # Move audio file
    os.rename(audio_file, audio_path)

    # Save frames
    for i, frame in enumerate(frames):
        cv2.imwrite(os.path.join(session_folder, f"frame_{i:04d}.jpg"), frame)

    print(f"📁 Saved all output → {session_folder}")
    try:
        post_to_supabase(report_path, audio_path)
    except Exception as e:
        print(f"⚠️ Error posting to Supabase: {e}")

    return report_path


# -----------------------------
# MAIN LOOP
# -----------------------------
def main():
    print("1")
    
    global oled
    print("2")

    oled = OLED_1in51()
    print("3")

    oled.Init()
    print("4")

    print("🔵 please STARTED")

    # Create output folder
    session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_folder = f"output/session_{session_ts}"
    os.makedirs(session_folder, exist_ok=True)

    print("📸 Using rpicam-still capture mode (no /dev/video0 required)")

    audio_filename = f"audio_{session_ts}.wav"

    audio_rec = AudioRecorder()
    audio_rec.start()

    frames_collected = []
    events = []

    print("🎥 Monitoring started... Press Q to quit.")

    last_audio_time = 0
    last_frame_time = 0
    session_start = time.time()

    try:
        while True:
            # ---- CAPTURE FRAME FROM PI CAMERA ----
            frame = capture_frame()
            if frame is None:
                print("⚠️ Frame capture failed, retrying...")
                continue

            frames_collected.append(frame.copy())
            now = time.time()

            # ---- AUDIO SAMPLE EVERY 5s ----
            if now - last_audio_time > 5:
                chunk_file = "temp_audio.wav"
                audio_rec.save_chunk(chunk_file)

                text = transcribe_chunk(chunk_file)
                if text:
                    audio_context.append(text)
                    if detect_shutdown_command(text):
                        oled_print("Voice shutdown command detected!\nEnding session safely...")
                        break
                    issue = detect_audio_keywords(text)
                    if issue:
                        print(f"🚨 AUDIO FLAG: {issue}")
                last_audio_time = now

            # ---- FRAME ANALYSIS EVERY 1.5s ----
            if now - last_frame_time > 1.5:
                result = analyze_frame(frame)
                result["timestamp"] = now - session_start
                events.append(result)

                visual_context.append(result["description"])
                if len(visual_context) > 10:
                    visual_context.pop(0)

                print("\n--- FRAME EVENT ---")
                print(result)
                realtime_routing_alert(result)
                last_frame_time = now

            # Manual shutdown
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("s"):
                print("\n🛑 Manual shutdown triggered.")
                break

    except KeyboardInterrupt:
        pass

    print("\n🛑 Ending session...")

    audio_rec.stop_and_save_full_audio(audio_filename)

    print("🎤 Transcribing full session audio...")
    full_audio_text = transcribe_chunk(audio_filename)

    print("📄 Generating final report...")
    report = generate_report(events, full_audio_text)

    save_output(report, audio_filename, frames_collected, session_folder)

    print("\n✅ Session complete.")

if __name__ == "__main__":
    print("yooo")
    main()
    print("noooo")
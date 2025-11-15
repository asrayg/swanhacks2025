import cv2
import base64
import time
from openai import OpenAI
from dotenv import load_dotenv
import os
from datetime import datetime
import pyaudio
import wave
import threading

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Audio recording settings
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
                data = self.stream.read(CHUNK)
                self.frames.append(data)
        
        self.thread = threading.Thread(target=record)
        self.thread.start()
        print("🎤 Audio recording started")
    
    def stop_recording(self):
        self.is_recording = False
        self.thread.join()
        self.stream.stop_stream()
        self.stream.close()
        
        # Save audio file
        wf = wave.open(self.filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.audio.get_sample_size(AUDIO_FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        self.audio.terminate()
        print(f"🎤 Audio saved to: {self.filename}")

def frame_to_base64(frame):
    frame_small = cv2.resize(frame, (320, 180))
    _, jpeg = cv2.imencode(".jpg", frame_small, [cv2.IMWRITE_JPEG_QUALITY, 30])
    return base64.b64encode(jpeg.tobytes()).decode("utf-8")

def analyze_frame(frame):
    img_b64 = frame_to_base64(frame)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
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

def transcribe_audio(audio_filename):
    """Transcribe audio file using Whisper API"""
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

def generate_clinical_report(summaries, audio_transcript, vital_signs):
    """Generate a clinical report using GPT"""
    # Combine all frame descriptions
    observations = "\n".join([f"At {ts:.1f}s: {desc}" for ts, desc in summaries])
    
    prompt = f"""You are a medical professional creating a clinical examination report. 

PATIENT VITAL SIGNS:
- Heart Rate: {vital_signs['heart_rate']}
- Blood Oxygen Level: {vital_signs['oxygen_level']}
- Height: {vital_signs['height']}

VISUAL OBSERVATIONS FROM EXAMINATION:
{observations}

AUDIO TRANSCRIPT FROM EXAMINATION:
{audio_transcript if audio_transcript else "No audio transcript available"}

Please create a professional clinical report that includes:
1. Patient Vital Signs (use the provided data)
2. Physical Examination Findings (based on visual observations)
3. Patient Communication (based on audio transcript)
4. Clinical Assessment
5. Recommendations

Format this as a formal medical report."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a medical professional creating clinical reports."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content

def save_clinical_report(report, audio_filename, filename=None):
    """Save the clinical report to a file"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"clinical_report_{timestamp}.txt"
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("CLINICAL EXAMINATION REPORT\n")
        f.write("=" * 70 + "\n")
        f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Audio Recording: {audio_filename}\n")
        f.write("=" * 70 + "\n\n")
        f.write(report)
        f.write("\n\n" + "=" * 70 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 70 + "\n")
    
    print(f"📋 Clinical report saved to: {filename}")
    return filename

def main():
    # Patient vital signs
    vital_signs = {
        "heart_rate": "85 bpm",
        "oxygen_level": "95%",
        "height": "5'10\""
    }
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return
    
    # Setup audio recording
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_filename = f"exam_audio_{timestamp}.wav"
    audio_recorder = AudioRecorder(audio_filename)
    
    print("🎥 Webcam started. Press q to quit.\n")
    print("🎤 Audio recording will start automatically.\n")
    
    summaries = []
    last_time = 0
    session_start = time.time()
    
    # Start audio recording
    audio_recorder.start_recording()
    
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
                elapsed = now - session_start
                
                print(f"\n🤖 GPT ({duration:.2f}s): {desc}")
                summaries.append((elapsed, desc))
                last_time = now
            
            cv2.imshow("Live Webcam (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
                
    except KeyboardInterrupt:
        pass
    
    print("\n🛑 Stopping examination...")
    
    # Stop audio recording
    audio_recorder.stop_recording()
    
    # Clean up video
    cap.release()
    cv2.destroyAllWindows()
    
    if not summaries:
        print("⚠️ No observations recorded.")
        return
    
    # Transcribe audio
    print("\n🎤 Transcribing audio...")
    audio_transcript = transcribe_audio(audio_filename)
    
    # Generate clinical report
    print("\n📋 Generating clinical report...")
    report = generate_clinical_report(summaries, audio_transcript, vital_signs)
    
    # Save clinical report
    report_filename = save_clinical_report(report, audio_filename)
    
    print("\n✅ Examination complete!")
    print(f"📁 Audio file: {audio_filename}")
    print(f"📁 Report file: {report_filename}")

if __name__ == "__main__":
    main()
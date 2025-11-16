import sounddevice as sd
import numpy as np
import wave
import time
from openai import OpenAI
from dotenv import load_dotenv
import os
from datetime import datetime
import tempfile
import subprocess

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class SpanishEnglishTranslator:
    def __init__(self, auto_calibrate=True):
        self.is_running = False
        self.sample_rate = 16000
        self.channels = 1
        self.silence_threshold = 10000  # Energy threshold for silence
        self.silence_duration = 1.5  # Seconds of silence before stopping
        self.auto_calibrate = auto_calibrate
        self.calibrated = False
        
    def record_with_silence_detection(self, max_duration=30):
        """Record audio until silence is detected"""
        chunk_duration = 0.5  # 500ms chunks
        
        # Auto-calibrate noise floor on first use
        if self.auto_calibrate and not self.calibrated:
            print("🔧 Calibrating noise floor... (stay quiet for 2 seconds)", end="\r")
            calibration_samples = []
            for _ in range(4):  # 4 chunks = 2 seconds
                chunk_data = sd.rec(
                    int(chunk_duration * self.sample_rate),
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype='int16'
                )
                sd.wait()
                energy = np.abs(chunk_data).mean()
                calibration_samples.append(energy)
            
            # Set threshold as 2.5x the average background noise
            avg_noise = np.mean(calibration_samples)
            self.silence_threshold = avg_noise
            self.calibrated = True
            print(f"✅ Noise floor: {avg_noise:.0f}, Speech threshold: {self.silence_threshold:.0f}")
        
        print("🎤 Listening... Speak now!", end="\r")
        
        # Storage for audio data
        all_audio_chunks = []
        silence_time = 0
        has_speech = False
        total_time = 0
        
        while total_time < max_duration and self.is_running:
            # Record a chunk
            chunk_data = sd.rec(
                int(chunk_duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='int16'
            )
            sd.wait()  # Wait for chunk to complete
            
            all_audio_chunks.append(chunk_data.copy())
            total_time += chunk_duration
            
            # Calculate energy (volume) of chunk
            energy = np.abs(chunk_data).mean()
            
            if energy > self.silence_threshold:
                # Speech detected
                silence_time = 0
                has_speech = True
                print("🎤 Listening... (speaking detected)", end="\r")
            else:
                # Silence detected
                if has_speech:
                    silence_time += chunk_duration
                    print(f"🎤 Listening... (silence {silence_time:.1f}s)", end="\r")
            
            # Stop if we've had enough silence after speech
            if has_speech and silence_time >= self.silence_duration:
                print("\n✅ Speech ended, processing...       ")
                break
        
        if not has_speech:
            print("\n⚠️  No speech detected                ")
            return None
        
        # Concatenate all audio chunks
        audio_data = np.concatenate(all_audio_chunks, axis=0)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_filename = temp_file.name
            
        with wave.open(temp_filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 2 bytes for int16
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data.tobytes())
        
        return temp_filename
    
    def transcribe_audio(self, audio_file):
        """Transcribe audio and detect language"""
        try:
            with open(audio_file, "rb") as f:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    response_format="verbose_json"
                )
            
            text = transcript.text.strip()
            language = transcript.language
            
            return text, language
            
        except Exception as e:
            print(f"❌ Error in transcription: {e}")
            return None, None
        finally:
            # Clean up temp file
            try:
                os.remove(audio_file)
            except:
                pass
    
    def detect_language(self, text, detected_lang):
        """
        Determine the language of the text.
        Returns: 'spanish' or 'english'
        """
        # Primary: Trust Whisper's language detection
        if detected_lang:
            if detected_lang == "es":
                return "spanish"
            elif detected_lang == "en":
                return "english"
        
        # Fallback: Check for Spanish-specific indicators
        spanish_chars = ['ñ', 'á', 'é', 'í', 'ó', 'ú', '¿', '¡']
        if any(char in text.lower() for char in spanish_chars):
            return "spanish"
        
        # Common Spanish words
        spanish_words = ['hola', 'gracias', 'por favor', 'buenos', 'días',
                        'cómo', 'qué', 'dónde', 'está', 'soy', 'muy',
                        'bien', 'señor', 'señora', 'buenas', 'noches']
        text_lower = text.lower()
        spanish_word_count = sum(1 for word in spanish_words if word in text_lower)
        
        # If multiple Spanish words found, it's likely Spanish
        if spanish_word_count >= 2:
            return "spanish"
        
        # Default to English if unclear
        return "english"
    
    def translate_to_english(self, spanish_text):
        """Translate Spanish to English"""
        try:
            translation = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a Spanish to English translator. Translate the following Spanish text to English. Only provide the translation, no explanations."
                    },
                    {
                        "role": "user",
                        "content": spanish_text
                    }
                ]
            )
            return translation.choices[0].message.content.strip()
        except Exception as e:
            print(f"❌ Error in Spanish→English translation: {e}")
            return None
    
    def translate_to_spanish(self, english_text):
        """Translate English to Spanish"""
        try:
            translation = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an English to Spanish translator. Translate the following English text to Spanish. Only provide the translation, no explanations."
                    },
                    {
                        "role": "user",
                        "content": english_text
                    }
                ]
            )
            return translation.choices[0].message.content.strip()
        except Exception as e:
            print(f"❌ Error in English→Spanish translation: {e}")
            return None
        
    def speak_text(self, text):
        """Generate and play speech"""
        try:
            response = client.audio.speech.create(
                model="tts-1-hd",
                voice="alloy",
                input=text,
                speed=0.75  # Slow down speech (0.25 to 4.0, default is 1.0)
            )
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                temp_file.write(response.content)
                speech_file = temp_file.name
            
            # Play audio using available audio player
            try:
                # Try mpv (common on Linux)
                subprocess.run(
                    ["mpv", "--really-quiet", speech_file],
                    check=True,
                    capture_output=True
                )
            except (FileNotFoundError, subprocess.CalledProcessError):
                try:
                    # Try ffplay (from ffmpeg)
                    subprocess.run(
                        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", speech_file],
                        check=True,
                        capture_output=True
                    )
                except (FileNotFoundError, subprocess.CalledProcessError):
                    try:
                        # Try paplay (PulseAudio)
                        subprocess.run(
                            ["paplay", speech_file],
                            check=True,
                            capture_output=True
                        )
                    except (FileNotFoundError, subprocess.CalledProcessError):
                        print("⚠️  No audio player found (mpv, ffplay, or paplay)")
            
            # Clean up
            try:
                os.remove(speech_file)
            except:
                pass
                
        except Exception as e:
            print(f"❌ Error in TTS: {e}")
    
    def listen_loop(self):
        """Main listening loop"""
        while self.is_running:
            try:
                # Record audio with silence detection
                audio_file = self.record_with_silence_detection()
                
                if not self.is_running:
                    break
                
                if not audio_file:
                    time.sleep(0.5)
                    continue
                
                print("⚙️  Transcribing...", end="\r")
                
                # Transcribe
                text, detected_lang = self.transcribe_audio(audio_file)
                
                if not text:
                    print("🟢 Ready for next input...   \n")
                    continue
                
                timestamp = datetime.now().strftime("%H:%M:%S")
                
                # Detect language and translate accordingly
                language = self.detect_language(text, detected_lang)
                
                if language == "spanish":
                    # Spanish → English
                    print(f"\n{'='*60}")
                    print(f"[{timestamp}] 🇪🇸 SPANISH DETECTED:")
                    print(f"  {text}")
                    print(f"  (Detected: {detected_lang})")
                    
                    english_translation = self.translate_to_english(text)
                    
                    if english_translation:
                        print(f"\n[{timestamp}] 🇬🇧 ENGLISH TRANSLATION:")
                        print(f"  {english_translation}")
                        print(f"\n🔊 Speaking in English...")
                        self.speak_text(english_translation)
                        print(f"✅ TTS completed")
                    print(f"{'='*60}")
                    
                else:  # English
                    # English → Spanish
                    print(f"\n{'='*60}")
                    print(f"[{timestamp}] 🇬🇧 ENGLISH DETECTED:")
                    print(f"  {text}")
                    print(f"  (Detected: {detected_lang})")
                    
                    spanish_translation = self.translate_to_spanish(text)
                    
                    if spanish_translation:
                        print(f"\n[{timestamp}] 🇪🇸 SPANISH TRANSLATION:")
                        print(f"  {spanish_translation}")
                        
                        print(f"\n🔊 Speaking in Spanish...")
                        self.speak_text(spanish_translation)
                        print(f"✅ TTS completed")
                    print(f"{'='*60}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                time.sleep(1)
    
    def start(self):
        """Start the translator"""
        self.is_running = True
        
        print("\n" + "="*60)
        print("🇪🇸 ↔️ 🇬🇧 SPANISH-ENGLISH LIVE TRANSLATOR")
        print("="*60)
        print("\nHow it works:")
        print("  • Automatically detects Spanish or English")
        print("  • Spanish → English translation + TTS")
        print("  • English → Spanish translation + TTS")
        print("  • Automatic silence detection (stops after 1.5s)")
        print("\nPress Ctrl+C to stop\n")
        print("="*60 + "\n")
        
        try:
            self.listen_loop()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
    
    def stop(self):
        """Stop the translator"""
        print("\n\n🛑 Translator stopped.")
        self.is_running = False

def main():
    translator = SpanishEnglishTranslator()
    translator.start()

if __name__ == "__main__":
    main()
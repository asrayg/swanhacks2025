import pyaudio
import wave
import time
from openai import OpenAI
from dotenv import load_dotenv
import os
from datetime import datetime

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 3  # Record in 3-second chunks

class HindiEnglishTranslator:
    def __init__(self):
        self.is_running = False
        self.audio = pyaudio.PyAudio()
        
    def record_audio_chunk(self):
        """Record a short audio chunk"""
        stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        
        frames = []
        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            if not self.is_running:
                break
            data = stream.read(CHUNK)
            frames.append(data)
        
        stream.stop_stream()
        stream.close()
        
        # Save to temp file
        temp_filename = f"temp_audio_{int(time.time())}.wav"
        wf = wave.open(temp_filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        
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
    
    def is_hindi(self, text, detected_lang):
        """Determine if text is Hindi"""
        # Check Whisper's language detection
        if detected_lang == "hi":
            return True
        
        # Check for common Hindi words (in case Whisper mislabels)
        hindi_words = ['namaste', 'namaskar', 'kaise', 'aap', 'hai', 'hain', 
                       'kya', 'main', 'mera', 'mere', 'tumhara', 'apka']
        if any(word in text.lower() for word in hindi_words):
            return True
        
        # Check for Devanagari script
        if any('\u0900' <= char <= '\u097F' for char in text):
            return True
        
        return False
    
    def translate_to_english(self, hindi_text):
        """Translate Hindi to English"""
        try:
            translation = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a Hindi to English translator. Translate the following Hindi text to English. Only provide the translation, no explanations."
                    },
                    {
                        "role": "user",
                        "content": hindi_text
                    }
                ]
            )
            return translation.choices[0].message.content.strip()
        except Exception as e:
            print(f"❌ Error in Hindi→English translation: {e}")
            return None
    
    def translate_to_hindi(self, english_text):
        """Translate English to Hindi"""
        try:
            translation = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an English to Hindi translator. Translate the following English text to Hindi. Only provide the translation in Devanagari script, no explanations."
                    },
                    {
                        "role": "user",
                        "content": english_text
                    }
                ]
            )
            return translation.choices[0].message.content.strip()
        except Exception as e:
            print(f"❌ Error in English→Hindi translation: {e}")
            return None
    
    def speak_text(self, text):
        """Generate and play speech"""
        try:
            response = client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=text
            )
            
            speech_file = f"speech_{int(time.time())}.mp3"
            response.stream_to_file(speech_file)
            
            # Play audio (using system command)
            if os.name == 'posix':  # macOS/Linux
                os.system(f"afplay {speech_file} 2>/dev/null")
            elif os.name == 'nt':  # Windows
                os.system(f"start {speech_file}")
            
            # Clean up
            time.sleep(2)
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
                print("\n🔴 Recording...", end="\r")
                
                # Record audio chunk
                audio_file = self.record_audio_chunk()
                
                if not self.is_running:
                    break
                
                print("⚙️  Processing...", end="\r")
                
                # Transcribe
                text, detected_lang = self.transcribe_audio(audio_file)
                
                if not text:
                    print("🟢 Listening...   ", end="\r")
                    continue
                
                timestamp = datetime.now().strftime("%H:%M:%S")
                
                # Determine if Hindi or English
                if self.is_hindi(text, detected_lang):
                    # Hindi → English
                    print(f"\n{'='*60}")
                    print(f"[{timestamp}] 🇮🇳 HINDI INPUT:")
                    print(f"  {text}")
                    
                    english_translation = self.translate_to_english(text)
                    
                    if english_translation:
                        print(f"\n[{timestamp}] 🇬🇧 ENGLISH TRANSLATION:")
                        print(f"  {english_translation}")
                    print(f"{'='*60}")
                    
                else:
                    # English → Hindi
                    print(f"\n{'='*60}")
                    print(f"[{timestamp}] 🇬🇧 ENGLISH INPUT:")
                    print(f"  {text}")
                    
                    hindi_translation = self.translate_to_hindi(text)
                    
                    if hindi_translation:
                        print(f"\n[{timestamp}] 🇮🇳 HINDI TRANSLATION:")
                        print(f"  {hindi_translation}")
                        
                        print(f"\n🔊 Speaking in Hindi...")
                        self.speak_text(hindi_translation)
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
        print("🇮🇳 ↔️ 🇬🇧 HINDI-ENGLISH LIVE TRANSLATOR")
        print("="*60)
        print("\nHow it works:")
        print("  • Speak in Hindi → Shows English translation")
        print("  • Speak in English → Shows Hindi translation + TTS")
        print("\nPress Ctrl+C to stop\n")
        print("="*60)
        
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
        self.audio.terminate()

def main():
    translator = HindiEnglishTranslator()
    translator.start()

if __name__ == "__main__":
    main()
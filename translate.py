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
import threading
import base64

# OLED imports
try:
    import board
    import digitalio
    import busio
    from PIL import Image, ImageDraw, ImageFont
    OLED_AVAILABLE = True
except ImportError:
    OLED_AVAILABLE = False
    print("OLED libraries not available (board, busio, PIL)")

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class OLED_1in51:
    def __init__(self):
        if not OLED_AVAILABLE:
            raise RuntimeError("OLED libraries not available")
        
        self.spi = busio.SPI(board.SCLK, MOSI=board.MOSI)
        
        self.dc = digitalio.DigitalInOut(board.D24)
        self.rst = digitalio.DigitalInOut(board.D25)
        self.cs = digitalio.DigitalInOut(board.D8)
        for pin in [self.dc, self.rst, self.cs]:
            pin.direction = digitalio.Direction.OUTPUT
        
        while not self.spi.try_lock():
            pass
        self.spi.configure(baudrate=8000000, phase=0, polarity=0)
        self.spi.unlock()
        
        self.width = 128
        self.height = 64
        
        self.reset()
        self.init_display()
    
    def command(self, cmd):
        self.dc.value = 0
        self.cs.value = 0
        self.spi.write(bytes([cmd]))
        self.cs.value = 1
    
    def data(self, data):
        self.dc.value = 1
        self.cs.value = 0
        self.spi.write(data)
        self.cs.value = 1
    
    def reset(self):
        self.rst.value = 0
        time.sleep(0.2)
        self.rst.value = 1
        time.sleep(0.2)
    
    def init_display(self):
        cmds = [
            0xAE, 0xD5, 0xA0, 0xA8, 0x3F, 0xD3, 0x00,
            0x40, 0xA1, 0xC8, 0xDA, 0x12, 0x81, 0x7F,
            0xA4, 0xA6, 0xD9, 0xF1, 0xDB, 0x40, 0xAF
        ]
        for c in cmds:
            self.command(c)
    
    def getbuffer(self, image):
        buf = [0x00] * (self.width * self.height // 8)
        img = image.convert("1")
        pixels = img.load()
        for y in range(self.height):
            for x in range(self.width):
                if pixels[x, y] == 0:
                    buf[x + (y // 8) * self.width] |= (1 << (y % 8))
        return buf
    
    def show_image(self, buf):
        for page in range(8):
            self.command(0xB0 + page)
            self.command(0x00)
            self.command(0x10)
            start = page * 128
            end = start + 128
            self.data(bytes(buf[start:end]))
    
    def clear(self):
        img = Image.new("1", (self.width, self.height), "white")
        buf = self.getbuffer(img)
        self.show_image(buf)
    
    def show_text(self, text, font_size=16, center=True):
        img = Image.new("1", (self.width, self.height), "white")
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        lines = []
        words = text.split()
        current_line = ""
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            bbox = draw.textbbox((0, 0), test_line, font=font)
            if bbox[2] - bbox[0] <= self.width - 10:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        y_offset = 10
        for line in lines[:3]: 
            if center:
                bbox = draw.textbbox((0, 0), line, font=font)
                text_width = bbox[2] - bbox[0]
                x = (self.width - text_width) // 2
            else:
                x = 5
            draw.text((x, y_offset), line, fill="black", font=font)
            y_offset += font_size + 4
        img = img.rotate(180)
 
        buf = self.getbuffer(img)
        self.show_image(buf)


class SpanishEnglishTranslator:
    def __init__(self, silence_threshold=None, auto_calibrate=False, use_oled=True):
        self.is_running = False
        self.sample_rate = 16000
        self.channels = 1
        self.silence_threshold = silence_threshold if silence_threshold is not None else 3000
        self.silence_duration = 1.5 
        self.auto_calibrate = auto_calibrate
        self.calibrated = not auto_calibrate  
        
        self.oled = None
        if use_oled and OLED_AVAILABLE:
            try:
                self.oled = OLED_1in51()
                self.oled.clear()
                print("OLED display initialized")
            except Exception as e:
                print(f"OLED initialization failed: {e}")
                self.oled = None
        
    def record_with_silence_detection(self, max_duration=30):
        chunk_duration = 0.5  
        if self.auto_calibrate and not self.calibrated:
            print("Calibrating noise floor... (stay quiet for 2 seconds)", end="\r")
            calibration_samples = []
            for _ in range(4):
                chunk_data = sd.rec(
                    int(chunk_duration * self.sample_rate),
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype='int16'
                )
                sd.wait()
                energy = np.abs(chunk_data).mean()
                calibration_samples.append(energy)
            
            avg_noise = np.mean(calibration_samples)
            self.silence_threshold = avg_noise
            self.calibrated = True
            print(f"Noise floor: {avg_noise:.0f}, Speech threshold: {self.silence_threshold:.0f}")
        
        print("🎤 Listening... Speak now!", end="\r")
        
        all_audio_chunks = []
        silence_time = 0
        has_speech = False
        total_time = 0
        
        while total_time < max_duration and self.is_running:
            chunk_data = sd.rec(
                int(chunk_duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='int16'
            )
            sd.wait()
            
            all_audio_chunks.append(chunk_data.copy())
            total_time += chunk_duration
            
            energy = np.abs(chunk_data).mean()
            
            if energy > self.silence_threshold:
                silence_time = 0
                has_speech = True
                print("Listening... (speaking detected)", end="\r")
            else:
                if has_speech:
                    silence_time += chunk_duration
                    print(f"Listening... (silence {silence_time:.1f}s)", end="\r")
            
            if has_speech and silence_time >= self.silence_duration:
                print("\n Speech ended, processing...       ")
                break
        
        if not has_speech:
            print("\n No speech detected                ")
            return None
        
        audio_data = np.concatenate(all_audio_chunks, axis=0)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_filename = temp_file.name
            
        with wave.open(temp_filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2) 
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data.tobytes())
        
        return temp_filename
    
    def transcribe_audio(self, audio_file):
        try:
            with open(audio_file, "rb") as f:
                audio_data = base64.b64encode(f.read()).decode('utf-8')
            
            response = client.chat.completions.create(
                model="gpt-4o-transcribe",
                modalities=["text"],
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": audio_data,
                                    "format": "wav"
                                }
                            },
                            {
                                "type": "text",
                                "text": "Transcribe this audio exactly as spoken. Also identify if the language is English ('en') or Spanish ('es'). Respond in this format: LANGUAGE: [en or es]\nTRANSCRIPT: [exact transcription]"
                            }
                        ]
                    }
                ]
            )
            
            full_response = response.choices[0].message.content.strip()
            
            lines = full_response.split('\n')
            language = None
            text = None
            
            for line in lines:
                if line.startswith('LANGUAGE:'):
                    lang_code = line.replace('LANGUAGE:', '').strip().lower()
                    language = lang_code if lang_code in ['en', 'es'] else None
                elif line.startswith('TRANSCRIPT:'):
                    text = line.replace('TRANSCRIPT:', '').strip()
            
            if not text:
                text = full_response
            
            if not language:
                language = 'en'
            
            return text, language
            
        except Exception as e:
            print(f"Error in GPT-4o transcription: {e}")
            return None, None
        finally:
            try:
                os.remove(audio_file)
            except:
                pass
    
    def detect_language(self, text, detected_lang):
        if detected_lang:
            if detected_lang in ["es", "spanish"]:
                return "spanish"
            elif detected_lang in ["en", "english"]:
                return "english"
            else:
                print(f"Detected language '{detected_lang}' not supported. Mapping to English or Spanish...")
        
        spanish_chars = ['ñ', 'á', 'é', 'í', 'ó', 'ú', '¿', '¡']
        if any(char in text.lower() for char in spanish_chars):
            print("🔍 Spanish character detected, treating as Spanish")
            return "spanish"
        
        spanish_words = ['hola', 'gracias', 'por favor', 'buenos', 'días',
                        'cómo', 'qué', 'dónde', 'está', 'soy', 'muy',
                        'bien', 'señor', 'señora', 'buenas', 'noches', 'como']
        text_lower = text.lower()
        spanish_word_count = sum(1 for word in spanish_words if word in text_lower)
        
        if spanish_word_count >= 2:
            print(f"{spanish_word_count} Spanish words detected, treating as Spanish")
            return "spanish"
        
        print("Defaulting to English")
        return "english"
    
    def translate_to_english(self, spanish_text):
        """Translate Spanish to English"""
        try:
            translation = client.chat.completions.create(
                model="gpt-5.1",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a Spanish to English translator. You ONLY translate from Spanish to English. If the text is already in English or in any other language, translate it to English anyway. Provide only the translation, no explanations."
                    },
                    {
                        "role": "user",
                        "content": spanish_text
                    }
                ]
            )
            return translation.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error in Spanish→English translation: {e}")
            return None
    
    def translate_to_spanish(self, english_text):
        """Translate English to Spanish"""
        try:
            translation = client.chat.completions.create(
                model="gpt-5.1",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an English to Spanish translator. You ONLY translate from English to Spanish. If the text is already in Spanish or in any other language, translate it to Spanish anyway. Provide only the translation, no explanations."
                    },
                    {
                        "role": "user",
                        "content": english_text
                    }
                ]
            )
            return translation.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error in English→Spanish translation: {e}")
            return None
        
    def speak_text(self, text):
        """Generate and play speech with OLED scrolling text"""
        try:
            response = client.audio.speech.create(
                model="tts-1-hd",
                voice="alloy",
                input=text,
                speed=0.75 
            )
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                temp_file.write(response.content)
                speech_file = temp_file.name
            
            words = len(text.split())
            base_duration = (words / 150) * 60  
            adjusted_duration = base_duration / 0.75 
            if self.oled:
                audio_finished = threading.Event()
                
                def play_audio():
                    try:
                        subprocess.run(
                            ["mpv", "--really-quiet", speech_file],
                            check=True,
                            capture_output=True
                        )
                    except (FileNotFoundError, subprocess.CalledProcessError):
                        try:
                            subprocess.run(
                                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", speech_file],
                                check=True,
                                capture_output=True
                            )
                        except (FileNotFoundError, subprocess.CalledProcessError):
                            try:
                                subprocess.run(
                                    ["paplay", speech_file],
                                    check=True,
                                    capture_output=True
                                )
                            except (FileNotFoundError, subprocess.CalledProcessError):
                                pass
                    finally:
                        audio_finished.set()
                
                audio_thread = threading.Thread(target=play_audio, daemon=True)
                audio_thread.start()
                
                words_list = text.split()
                delay_per_word = adjusted_duration / len(words_list) if words_list else 0
                
                for i in range(len(words_list)):
                    start_idx = max(0, i - 1)
                    end_idx = min(len(words_list), i + 2)
                    display_text = " ".join(words_list[start_idx:end_idx])
                    
                    try:
                        self.oled.show_text(display_text, font_size=14, center=True)
                    except Exception as e:
                        print(f"OLED error: {e}")
                        break
                    
                    time.sleep(delay_per_word)
                
                audio_finished.wait(timeout=adjusted_duration + 2)
                
                try:
                    self.oled.clear()
                except:
                    pass
            else:
                try:
                    subprocess.run(
                        ["mpv", "--really-quiet", speech_file],
                        check=True,
                        capture_output=True
                    )
                except (FileNotFoundError, subprocess.CalledProcessError):
                    try:
                        subprocess.run(
                            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", speech_file],
                            check=True,
                            capture_output=True
                        )
                    except (FileNotFoundError, subprocess.CalledProcessError):
                        try:
                            subprocess.run(
                                ["paplay", speech_file],
                                check=True,
                                capture_output=True
                            )
                        except (FileNotFoundError, subprocess.CalledProcessError):
                            print("No audio player found (mpv, ffplay, or paplay)")
            
            try:
                os.remove(speech_file)
            except:
                pass
                
        except Exception as e:
            print(f"Error in TTS: {e}")
    
    def listen_loop(self):
        while self.is_running:
            try:
                if self.oled:
                    try:
                        self.oled.show_text("LISTENING", font_size=18, center=True)
                    except:
                        pass
                
                audio_file = self.record_with_silence_detection()
                
                if not self.is_running:
                    break
                
                if not audio_file:
                    time.sleep(0.5)
                    continue
                
                if self.oled:
                    try:
                        self.oled.show_text("Transcribing...", font_size=16, center=True)
                    except:
                        pass
                
                print("Transcribing...", end="\r")
                
                text, detected_lang = self.transcribe_audio(audio_file)
                
                if not text:
                    print("Ready for next input...   \n")
                    if self.oled:
                        try:
                            self.oled.clear()
                        except:
                            pass
                    continue
                
                timestamp = datetime.now().strftime("%H:%M:%S")
                
                language = self.detect_language(text, detected_lang)
                
                if language == "english":
                    print(f"\n{'='*60}")
                    print(f"[{timestamp}] ENGLISH DETECTED:")
                    print(f"  {text}")
                    print(f"  (Detected: {detected_lang})")
                    
                    if self.oled:
                        try:
                            self.oled.show_text("Translating to Spanish...", font_size=14, center=True)
                        except:
                            pass
                    
                    spanish_translation = self.translate_to_spanish(text)
                    
                    if spanish_translation:
                        print(f"\n[{timestamp}] 🇪🇸 SPANISH TRANSLATION:")
                        print(f"  {spanish_translation}")
                        
                        print(f"\nSpeaking in Spanish...")
                        self.speak_text(spanish_translation)
                        print(f"TTS completed")
                    print(f"{'='*60}")
                
                   
                    
                else: 
                    print(f"\n{'='*60}")
                    print(f"[{timestamp}] SPANISH DETECTED:")
                    print(f"  {text}")
                    print(f"  (Detected: {detected_lang})")
                    
                    if self.oled:
                        try:
                            self.oled.show_text("Translating to English...", font_size=14, center=True)
                        except:
                            pass
                    
                    english_translation = self.translate_to_english(text)
                    
                    if english_translation:
                        print(f"\n[{timestamp}] ENGLISH TRANSLATION:")
                        print(f"  {english_translation}")
                        print(f"\n Speaking in English...")
                        self.speak_text(english_translation)
                        print(f" TTS completed")
                    print(f"{'='*60}")
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\n Error: {e}")
                time.sleep(1)
    
    def start(self):
        self.is_running = True
        
        print("\n" + "="*60)
        print("SPANISH-ENGLISH LIVE TRANSLATOR")
        print("="*60)
        print("\nPowered by GPT-4o Audio")
        print("Supported Languages: ENGLISH and SPANISH ONLY")
        print("\nHow it works:")
        print("  • GPT-4o transcribes audio and detects language")
        print("  • Spanish → English translation + TTS")
        print("  • English → Spanish translation + TTS")
        print("  • Automatic silence detection (stops after 1.5s)")
        print("\nPress Ctrl+C to stop\n")
        print("="*60 + "\n")
        
        if self.oled:
            try:
                self.oled.show_text("TRANSLATOR READY", font_size=16, center=True)
                time.sleep(1)
            except:
                pass
        
        try:
            self.listen_loop()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
    
    def stop(self):
        """Stop the translator"""
        print("\n\n Translator stopped.")
        self.is_running = False
        
        if self.oled:
            try:
                self.oled.clear()
            except:
                pass

def main():
    translator = SpanishEnglishTranslator()
    translator.start()

if __name__ == "__main__":
    main()
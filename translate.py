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

# OLED Display imports
try:
    import board
    import digitalio
    import busio
    from PIL import Image, ImageDraw, ImageFont
    OLED_AVAILABLE = True
except ImportError:
    OLED_AVAILABLE = False
    print("⚠️ OLED libraries not available. Display features disabled.")

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# OLED Constants
OLED_WIDTH = 128
OLED_HEIGHT = 64


class OLED_1in51:
    """OLED Display controller for 128x64 SSD1306-based displays"""
    
    def __init__(self):
        if not OLED_AVAILABLE:
            raise RuntimeError("OLED libraries not available")
        
        # SPI
        self.spi = busio.SPI(board.SCLK, MOSI=board.MOSI)
        
        # Pins
        self.dc = digitalio.DigitalInOut(board.D24)
        self.rst = digitalio.DigitalInOut(board.D25)
        self.cs = digitalio.DigitalInOut(board.D8)
        for pin in [self.dc, self.rst, self.cs]:
            pin.direction = digitalio.Direction.OUTPUT
        
        # Configure SPI
        while not self.spi.try_lock():
            pass
        self.spi.configure(baudrate=8000000, phase=0, polarity=0)
        self.spi.unlock()
        
        self.width = OLED_WIDTH
        self.height = OLED_HEIGHT
        
        print("✅ OLED Display initialized")
    
    def command(self, cmd):
        """Send command to OLED"""
        self.dc.value = 0
        self.cs.value = 0
        self.spi.write(bytes([cmd]))
        self.cs.value = 1
    
    def data(self, data):
        """Send data to OLED"""
        self.dc.value = 1
        self.cs.value = 0
        self.spi.write(data)
        self.cs.value = 1
    
    def reset(self):
        """Hardware reset of OLED"""
        self.rst.value = 0
        time.sleep(0.2)
        self.rst.value = 1
        time.sleep(0.2)
    
    def init(self):
        """Initialize OLED with standard SSD1306 commands"""
        self.reset()
        cmds = [
            0xAE,  # Display OFF
            0xD5, 0xA0,  # Set display clock
            0xA8, 0x3F,  # Set multiplex ratio
            0xD3, 0x00,  # Set display offset
            0x40,  # Set start line
            0xA1,  # Set segment remap
            0xC8,  # Set COM scan direction
            0xDA, 0x12,  # Set COM pins
            0x81, 0x7F,  # Set contrast
            0xA4,  # Display follows RAM
            0xA6,  # Normal display (not inverted)
            0xD9, 0xF1,  # Set pre-charge period
            0xDB, 0x40,  # Set VCOMH deselect level
            0xAF  # Display ON
        ]
        for c in cmds:
            self.command(c)
    
    def getbuffer(self, image):
        """Convert PIL image to OLED buffer format"""
        buf = [0x00] * (self.width * self.height // 8)
        img = image.convert("1")
        pixels = img.load()
        for y in range(self.height):
            for x in range(self.width):
                if pixels[x, y] == 0:
                    buf[x + (y // 8) * self.width] |= (1 << (y % 8))
        return buf
    
    def show_image(self, buf):
        """Display buffer on OLED"""
        for page in range(8):
            self.command(0xB0 + page)  # Set page address
            self.command(0x00)  # Set lower column address
            self.command(0x10)  # Set higher column address
            start = page * 128
            end = start + 128
            self.data(bytes(buf[start:end]))
    
    def clear(self):
        """Clear the display"""
        img = Image.new("1", (self.width, self.height), "white")
        buf = self.getbuffer(img)
        self.show_image(buf)
    
    def scroll_text(self, text, font_size=16, scroll_speed=0.05):
        """
        Scroll text horizontally across the display.
        
        Args:
            text: Text to display
            font_size: Font size in pixels
            scroll_speed: Delay between scroll steps (seconds)
        """
        try:
            # Try to load a nice font
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            # Fall back to default font
            font = ImageFont.load_default()
        
        # Create a temporary image to measure text width
        temp_img = Image.new("1", (1, 1), "white")
        temp_draw = ImageDraw.Draw(temp_img)
        bbox = temp_draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Calculate vertical centering
        y_pos = (self.height - text_height) // 2
        
        # Scroll from right to left
        for x_offset in range(self.width, -text_width - 10, -2):
            img = Image.new("1", (self.width, self.height), "white")
            draw = ImageDraw.Draw(img)
            draw.text((x_offset, y_pos), text, fill="black", font=font)
            
            buf = self.getbuffer(img)
            self.show_image(buf)
            time.sleep(scroll_speed)
        
        # Clear display after scrolling
        self.clear()


class SpanishEnglishTranslator:
    def __init__(self):
        self.is_running = False
        self.sample_rate = 44100  # CD quality, standard for plughw
        self.channels = 2  # Stereo
        self.silence_threshold = 1500  # Energy threshold for silence
        self.silence_duration = 1.5  # Seconds of silence before stopping
        
        # Setup microphone device (plughw:2,0)
        self.mic_device = None
        self.mic_device_index = None
        self._setup_microphone()
        
        # Initialize OLED display if available
        self.oled = None
        if OLED_AVAILABLE:
            try:
                self.oled = OLED_1in51()
                self.oled.init()
                self.oled.clear()
                print("📺 OLED Display ready for translations")
            except Exception as e:
                print(f"⚠️ Could not initialize OLED: {e}")
                self.oled = None
        else:
            print("ℹ️ Running without OLED display")
    
    def list_audio_devices(self):
        """List all available audio devices"""
        print("\n📋 Available Audio Devices:")
        print("="*60)
        devices = sd.query_devices()
        for idx, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"  [{idx}] {device['name']}")
                print(f"      Input Channels: {device['max_input_channels']}")
                print(f"      Sample Rate: {device['default_samplerate']} Hz")
        print("="*60 + "\n")
    
    def _setup_microphone(self):
        """Setup microphone to use plughw:2,0 (hardware card 2, device 0)"""
        try:
            # Set ALSA device environment variable for direct hardware access
            os.environ['AUDIODEV'] = 'plughw:2,0'
            
            devices = sd.query_devices()
            
            print("\n🔍 Searching for microphone device plughw:2,0...")
            
            # Strategy 1: Look for exact ALSA device name patterns
            target_patterns = [
                'hw:2,0', 'plughw:2,0', 'hw:2', 'card 2'
            ]
            
            for idx, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    device_name = device['name'].lower()
                    if any(pattern in device_name for pattern in target_patterns):
                        self.mic_device_index = idx
                        self.mic_device = device
                        self.sample_rate = int(device['default_samplerate'])
                        print(f"✅ Found target device!")
                        print(f"   Device: {device['name']}")
                        print(f"   Index: {idx}")
                        print(f"   Sample rate: {self.sample_rate} Hz")
                        print(f"   Channels: {device['max_input_channels']}")
                        return
            
            # Strategy 2: Try PulseAudio device (which can route to plughw:2,0)
            print("⚠️ Direct ALSA device not found, trying PulseAudio...")
            for idx, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    if 'pulse' in device['name'].lower():
                        self.mic_device_index = idx
                        self.mic_device = device
                        self.sample_rate = int(device['default_samplerate'])
                        print(f"✅ Using PulseAudio (will route to plughw:2,0)")
                        print(f"   Device: {device['name']}")
                        print(f"   Index: {idx}")
                        print(f"   Sample rate: {self.sample_rate} Hz")
                        return
            
            # Strategy 3: Use default input device
            print("⚠️ Falling back to default input device")
            self.list_audio_devices()
            default_input = sd.default.device[0]
            if default_input is not None:
                self.mic_device_index = default_input
                self.mic_device = devices[default_input]
                self.sample_rate = int(self.mic_device['default_samplerate'])
                print(f"✅ Using default device: {self.mic_device['name']}")
            else:
                print("⚠️ Using system default (no specific device)")
                self.mic_device_index = None
            
        except Exception as e:
            print(f"❌ Error setting up microphone: {e}")
            import traceback
            traceback.print_exc()
            print("   Will use system default device")
            self.mic_device_index = None
        
    def record_with_silence_detection(self, max_duration=30):
        """Record audio until silence is detected with live visualization"""
        chunk_duration = 0.5  # 500ms chunks
        
        print("\n" + "="*70)
        print("🎤 LIVE AUDIO MONITORING")
        print("="*70)
        
        # Storage for audio data
        all_audio_chunks = []
        silence_time = 0
        has_speech = False
        total_time = 0
        
        # Determine actual channels to use
        actual_channels = self.channels
        if self.mic_device:
            actual_channels = min(self.channels, self.mic_device['max_input_channels'])
        
        # For live transcription attempts
        transcribe_interval = 2.0  # Try to transcribe every 2 seconds
        last_transcribe_time = 0
        partial_transcript = ""
        
        while total_time < max_duration and self.is_running:
            # Record a chunk
            chunk_data = sd.rec(
                int(chunk_duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=actual_channels,
                dtype='int16',
                device=self.mic_device_index
            )
            sd.wait()  # Wait for chunk to complete
            
            all_audio_chunks.append(chunk_data.copy())
            total_time += chunk_duration
            
            # Calculate energy (volume) of chunk
            energy = np.abs(chunk_data).mean()
            
            # Create visual audio level meter
            meter_width = 50
            energy_normalized = min(energy / 3000, 1.0)  # Normalize to 0-1
            filled_bars = int(energy_normalized * meter_width)
            meter = "█" * filled_bars + "░" * (meter_width - filled_bars)
            
            # Status display
            if energy > self.silence_threshold:
                # Speech detected
                silence_time = 0
                has_speech = True
                status = f"🔴 SPEAKING [{total_time:5.1f}s] {meter} {int(energy):5d}"
            else:
                # Silence detected
                if has_speech:
                    silence_time += chunk_duration
                    status = f"🟡 SILENCE  [{total_time:5.1f}s] {meter} {int(energy):5d} (pause: {silence_time:.1f}s)"
                else:
                    status = f"🟢 WAITING  [{total_time:5.1f}s] {meter} {int(energy):5d}"
            
            # Print status on same line
            print(f"\r{status}", end="", flush=True)
            
            # Attempt live transcription every N seconds
            if has_speech and (total_time - last_transcribe_time) >= transcribe_interval:
                last_transcribe_time = total_time
                try:
                    # Try to transcribe recent audio
                    self._attempt_live_transcription(all_audio_chunks, actual_channels)
                except:
                    pass  # Ignore transcription errors during live monitoring
            
            # Stop if we've had enough silence after speech
            if has_speech and silence_time >= self.silence_duration:
                print(f"\n{'='*70}")
                print("✅ Speech ended, processing final transcription...")
                break
        
        if not has_speech:
            print(f"\n{'='*70}")
            print("⚠️  No speech detected")
            print("="*70 + "\n")
            return None
        
        # Concatenate all audio chunks
        audio_data = np.concatenate(all_audio_chunks, axis=0)
        
        # Convert stereo to mono if needed (for Whisper compatibility)
        if actual_channels == 2 and len(audio_data.shape) > 1:
            # Average the two channels to create mono
            audio_data = audio_data.mean(axis=1).astype('int16')
            final_channels = 1
        else:
            final_channels = actual_channels
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_filename = temp_file.name
            
        with wave.open(temp_filename, 'wb') as wf:
            wf.setnchannels(final_channels)
            wf.setsampwidth(2)  # 2 bytes for int16
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data.tobytes())
        
        return temp_filename
    
    def _attempt_live_transcription(self, audio_chunks, channels):
        """Attempt to transcribe recent audio chunks and display live"""
        try:
            # Use last 3 seconds of audio for live transcription
            recent_duration = 3.0
            chunks_to_use = int(recent_duration / 0.5)
            recent_chunks = audio_chunks[-chunks_to_use:] if len(audio_chunks) > chunks_to_use else audio_chunks
            
            if not recent_chunks:
                return
            
            # Concatenate recent audio
            audio_data = np.concatenate(recent_chunks, axis=0)
            
            # Convert stereo to mono if needed
            if channels == 2 and len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1).astype('int16')
                final_channels = 1
            else:
                final_channels = channels
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_filename = temp_file.name
            
            with wave.open(temp_filename, 'wb') as wf:
                wf.setnchannels(final_channels)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_data.tobytes())
            
            # Quick transcription (no verbose mode to save time)
            with open(temp_filename, "rb") as f:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f
                )
            
            # Clean up temp file
            try:
                os.remove(temp_filename)
            except:
                pass
            
            if transcript.text.strip():
                # Display live transcript on a new line
                print(f"\n💬 Live: \"{transcript.text.strip()}\"")
                # Return cursor to status line
                print("", end="", flush=True)
        
        except Exception as e:
            # Silently fail for live transcription
            pass
    
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
            english_text = translation.choices[0].message.content.strip()
            
            # Display on OLED if available (in background thread)
            if self.oled:
                display_thread = threading.Thread(
                    target=self._display_translation,
                    args=(english_text,),
                    daemon=True
                )
                display_thread.start()
            
            return english_text
        except Exception as e:
            print(f"❌ Error in Spanish→English translation: {e}")
            return None
    
    def _display_translation(self, text):
        """Display translation on OLED screen (runs in background thread)"""
        try:
            if self.oled:
                print("📺 Displaying on OLED...")
                self.oled.scroll_text(text, font_size=18, scroll_speed=0.03)
        except Exception as e:
            print(f"⚠️ Error displaying on OLED: {e}")
    
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
                
                print("⚙️  Transcribing final audio...", end="\r")
                
                # Transcribe
                text, detected_lang = self.transcribe_audio(audio_file)
                
                if not text:
                    print("\n🟢 Ready for next input...\n")
                    continue
                
                timestamp = datetime.now().strftime("%H:%M:%S")
                
                # Detect language and translate accordingly
                language = self.detect_language(text, detected_lang)
                
                if language == "spanish":
                    # Spanish → English
                    print(f"\n{'='*70}")
                    print(f"📝 FINAL TRANSCRIPTION [{timestamp}]")
                    print(f"{'='*70}")
                    print(f"🇪🇸 SPANISH: {text}")
                    print(f"   (Whisper detected: {detected_lang})")
                    print(f"{'-'*70}")
                    print(f"🔄 Translating Spanish → English...")
                    
                    english_translation = self.translate_to_english(text)
                    
                    if english_translation:
                        print(f"🇬🇧 ENGLISH: {english_translation}")
                        print(f"{'-'*70}")
                        print(f"🔊 Speaking translation in English...")
                        self.speak_text(english_translation)
                        print(f"✅ Complete!")
                    print(f"{'='*70}\n")
                    
                else:  # English
                    # English → Spanish
                    print(f"\n{'='*70}")
                    print(f"📝 FINAL TRANSCRIPTION [{timestamp}]")
                    print(f"{'='*70}")
                    print(f"🇬🇧 ENGLISH: {text}")
                    print(f"   (Whisper detected: {detected_lang})")
                    print(f"{'-'*70}")
                    print(f"🔄 Translating English → Spanish...")
                    
                    spanish_translation = self.translate_to_spanish(text)
                    
                    if spanish_translation:
                        print(f"🇪🇸 SPANISH: {spanish_translation}")
                        print(f"{'-'*70}")
                        print(f"🔊 Speaking translation in Spanish...")
                        self.speak_text(spanish_translation)
                        print(f"✅ Complete!")
                    print(f"{'='*70}\n")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                time.sleep(1)
    
    def start(self):
        """Start the translator"""
        self.is_running = True
        
        print("\n" + "="*70)
        print("🇪🇸 ↔️ 🇬🇧 SPANISH-ENGLISH LIVE TRANSLATOR WITH REAL-TIME DISPLAY")
        print("="*70)
        print("\n✨ Features:")
        print("  • Live audio visualization with volume meter")
        print("  • Real-time word detection as you speak")
        print("  • Automatic language detection (Spanish/English)")
        print("  • Bidirectional translation with TTS")
        print("  • Auto-stops after 1.5s of silence")
        print("  • English translations scroll on OLED display")
        print("\n📊 Visual Indicators:")
        print("  🔴 SPEAKING - Active speech detected")
        print("  🟡 SILENCE  - Pause detected (counting down)")
        print("  🟢 WAITING  - Ready for input")
        print("\n💬 Live transcription updates every 2 seconds")
        print("\nPress Ctrl+C to stop")
        print("="*70 + "\n")
        
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
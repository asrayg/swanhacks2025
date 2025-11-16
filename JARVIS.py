"""
JARVIS - A modular AI assistant library with RAG capabilities
Can be imported and used in any Python project
"""

import os
import pathlib
import uuid
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
import chromadb
from chromadb.config import Settings
import PyPDF2
import sounddevice as sd
import numpy as np
import io
import wave
import subprocess
import tempfile
from PIL import Image, ImageDraw, ImageFont
import threading
from driver import OLED_1in51, OLED_WIDTH, OLED_HEIGHT
import numpy as np
import base64

oled = None

import time


def oled_listening_animation(stop_event):
    """Pulsing circle animation while JARVIS is listening for input."""
    global oled
    if oled is None:
        print("Listening animation: OLED is None, skipping animation")
        return
    
    print("Listening animation started")
    phase = 0
    frame_count = 0
    
    while not stop_event.is_set():
        image = Image.new("1", (OLED_WIDTH, OLED_HEIGHT), 255)
        draw = ImageDraw.Draw(image)

        center_x, center_y = OLED_WIDTH // 2, OLED_HEIGHT // 2
        pulse = abs(np.sin(phase)) 
        
        for i in range(3):
            radius = int(10 + i * 8 + pulse * 8)
            thickness = 2 if i == 1 else 1 
            
            bbox = [
                center_x - radius, center_y - radius,
                center_x + radius, center_y + radius
            ]
            draw.ellipse(bbox, outline=0, width=thickness)
        
        center_radius = int(3 + pulse * 2)
        center_bbox = [
            center_x - center_radius, center_y - center_radius,
            center_x + center_radius, center_y + center_radius
        ]
        draw.ellipse(center_bbox, fill=0)

        image = image.rotate(180)

        buf = oled.getbuffer(image)
        oled.ShowImage(buf)

        phase += 0.15
        frame_count += 1
        time.sleep(0.08)
    
    print(f"Listening animation stopped (rendered {frame_count} frames)")


def oled_speaking_animation(duration=3.0, speed=0.08):
    global oled
    if oled is None:
        return

    start = time.time()
    phase = 0

    while time.time() - start < duration:
        image = Image.new("1", (OLED_WIDTH, OLED_HEIGHT), 255)
        draw = ImageDraw.Draw(image)

        for x in range(0, OLED_WIDTH, 2):
            y = int(32 + 20 * np.sin((x / 10.0) + phase))
            draw.line((x, 32, x, y), fill=0)

        image = image.rotate(180)

        buf = oled.getbuffer(image)
        oled.ShowImage(buf)

        phase += 0.25
        time.sleep(speed)



class JARVIS:
    
    def __init__(
        self,
        api_key=None,
        model="gpt-5-nano",
        embedding_model="text-embedding-3-small",
        db_path=None,
        disable_telemetry=True,
        tts_voice="alloy",
        tts_model="tts-1-hd"
    ):

        if disable_telemetry:
            os.environ["ANONYMIZED_TELEMETRY"] = "False"
            os.environ["CHROMA_TELEMETRY_ENABLED"] = "False"
            os.environ["CHROMA_USE_OPENAI_EMBEDDINGS"] = "False"
            os.environ["CHROMA_DISABLE_TELEMETRY"] = "True"
        
        load_dotenv(find_dotenv())
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise RuntimeError(
                "OpenAI API key not found. Please provide via api_key parameter "
                "or set OPENAI_API_KEY environment variable."
            )
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.embedding_model = embedding_model
        
        self.db_path = pathlib.Path(db_path or "./jarvis_db")
        self.db_path.mkdir(exist_ok=True)
        
        settings = Settings(
            persist_directory=str(self.db_path),
            anonymized_telemetry=False,
            is_persistent=True
        )
        self.chroma_client = chromadb.Client(settings)
        
        self.collection_name = f"jarvis_memory_{uuid.uuid4().hex[:8]}"
        self.collection = self.chroma_client.create_collection(
            name=self.collection_name
        )
        
        self.context_chunks = []
        self.large_context = ""
        
        self.mic_device_index = None
        self.mic_device = None
        self.silence_threshold = 3000  
        self.tts_voice = tts_voice
        self.tts_model = tts_model 
        
        self.system_instructions = self._load_instructions()
        
        print(f"JARVIS initialized with collection: {self.collection_name}")
    
    def _load_instructions(self):
        if pathlib.Path("jarvis_instructions.txt").exists():
            try:
                print("Loaded system instructions")
                return pathlib.Path("jarvis_instructions.txt").read_text(encoding="utf-8")
            except Exception as e:
                print(f"Could not load instructions: {e}")
        
        return """You are JARVIS, an intelligent AI assistant. 
Be helpful, accurate, and concise. Use provided context when available."""
    
    def set_silence_threshold(self, threshold):
        self.silence_threshold = threshold
        print(f"JARVIS silence threshold set to: {threshold:.0f}")
    
    def reload_instructions(self, instructions_path=None):
        if instructions_path:
            instructions_file = pathlib.Path(instructions_path)
        else:
            instructions_file = pathlib.Path("jarvis_instructions.txt")
        
        if not instructions_file.exists():
            print(f"❌ Instructions file not found: {instructions_file}")
            return False
        
        try:
            self.system_instructions = instructions_file.read_text(encoding="utf-8")
            print(f"Reloaded instructions from {instructions_file}")
            return True
        except Exception as e:
            print(f"Error reloading instructions: {e}")
            return False
    
    def _embed_documents(self, texts):
        return [d.embedding for d in self.client.embeddings.create(
            model=self.embedding_model,
            input=texts
        ).data]
    
    def _embed_query(self, text):
        return self.client.embeddings.create(
            model=self.embedding_model,
            input=[text]
        ).data[0].embedding
    
    def _extract_text_from_file(self, filepath):
        if filepath.suffix.lower() == ".txt":
            return filepath.read_text(encoding="utf-8", errors="ignore").strip()
        elif filepath.suffix.lower() == ".pdf":
            try:
                reader = PyPDF2.PdfReader(str(filepath))
                return "\n".join(
                    page.extract_text() or "" for page in reader.pages
                ).strip()
            except Exception as e:
                print(f"⚠️ Could not read {filepath.name}: {e}")
        
        return ""
    
    def add_context_from_file(
        self,
        filepath,
        chunk_size=2000,
        chunk_overlap=200
    ):
        filepath = pathlib.Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        text = self._extract_text_from_file(filepath)
        
        if not text:
            print(f"No text extracted from {filepath.name}")
            return 0
        
        self.large_context += "\n\n" + text
        
        chunks = []
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunk = text[i:i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        
        if chunks:
            vectors = self._embed_documents(chunks)
            ids = [f"{filepath.stem}_{uuid.uuid4().hex[:6]}_{i}" 
                   for i in range(len(chunks))]
            
            self.collection.add(
                documents=chunks,
                embeddings=vectors,
                ids=ids,
                metadatas=[{"source": filepath.name} for _ in chunks]
            )
            
            self.context_chunks.extend(chunks)
            print(f"Added {len(chunks)} chunks from {filepath.name}")
        
        return len(chunks)
    
    def add_context_from_directory(
        self,
        directory,
        chunk_size=2000,
        chunk_overlap=200,
        recursive=False
    ):
        directory = pathlib.Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        total_chunks = 0
        file_count = 0
        
        if recursive:
            pattern = "**/*"
            print(f"Recursively searching {directory} for documents...")
        else:
            pattern = "*"
        
        for file in directory.glob(pattern):
            if file.is_file() and file.suffix.lower() in [".txt", ".pdf"]:
                try:
                    chunks = self.add_context_from_file(
                        str(file),
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    total_chunks += chunks
                    file_count += 1
                except Exception as e:
                    print(f"Error processing {file.name}: {e}")
        
        print(f"Total: {total_chunks} chunks from {file_count} files")
        return total_chunks
    
    def add_context_from_text(
        self,
        text,
        chunk_size=2000,
        chunk_overlap=200
    ):
        if not text.strip():
            return 0
        
        self.large_context += "\n\n" + text
        
        chunks = []
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunk = text[i:i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        
        if chunks:
            vectors = self._embed_documents(chunks)
            ids = [f"text_{uuid.uuid4().hex[:6]}_{i}" 
                   for i in range(len(chunks))]
            
            self.collection.add(
                documents=chunks,
                embeddings=vectors,
                ids=ids,
                metadatas=[{"source": "direct_text"} for _ in chunks]
            )
            
            self.context_chunks.extend(chunks)
            print(f"Added {len(chunks)} chunks from text")
        
        return len(chunks)
    
    def _retrieve_relevant_context(
        self,
        query,
        n_results=5
    ):
        try:
            return self.collection.query(
                query_embeddings=[self._embed_query(query)],
                n_results=n_results
            ).get("documents", [[]])[0]
        except Exception as e:
            print(f"Error retrieving context: {e}")
            return []
    
    def _transcribe_with_gpt4o(self, audio_file_path, language="en-US"):
        """
        Transcribe audio using GPT-4o-transcribe model
        Args:
            audio_file_path: Path to the WAV audio file
            language: Language code (e.g., "en-US") - passed for compatibility but not used by GPT-4o
        Returns:
            Transcribed text or None if failed
        """
        try:
            with open(audio_file_path, "rb") as f:
                audio_data = base64.b64encode(f.read()).decode('utf-8')
            
            response = self.client.chat.completions.create(
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
                                "text": "Transcribe this audio exactly as spoken."
                            }
                        ]
                    }
                ]
            )
            
            transcription = response.choices[0].message.content.strip()
            return transcription
            
        except Exception as e:
            print(f"Error in GPT-4o transcription: {e}")
            return None
    
    def ask(
        self,
        query,
        use_rag=True,
        n_results=5,
        include_sources=False,
        system_prompt=None
    ):
        context_parts = []
        sources = []
        
        if use_rag and self.context_chunks:
            relevant_docs = self._retrieve_relevant_context(query, n_results)
            if relevant_docs:
                context_parts.append("Relevant information:")
                context_parts.extend(relevant_docs)
                sources = ["Vector Database"] * len(relevant_docs)
        
        messages = [{"role": "system", "content": system_prompt or self.system_instructions}]
        
        if context_parts:
            messages.append({
                "role": "user",
                "content": f"Context:\n{'\n\n'.join(context_parts)}\n\nQuestion: {query}"
            })
        else:
            messages.append({"role": "user", "content": query})
        
        try:
            answer = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            ).choices[0].message.content
            
            if include_sources and sources:
                answer += f"\n\n[Sources: {', '.join(set(sources))}]"
            
            return answer
        
        except Exception as e:
            return f"Error generating response: {e}"
    
    def chat(
        self,
        messages,
        use_context=True,
        n_results=3
    ):
        if not messages:
            return "No messages provided."
        
        if use_context and self.context_chunks:
            relevant_docs = self._retrieve_relevant_context(messages[-1]["content"], n_results)
            if relevant_docs:
                messages.insert(-1, {"role": "system", "content": "Relevant context:\n" + "\n\n".join(relevant_docs)})
        
        try:
            return self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            ).choices[0].message.content
        
        except Exception as e:
            return f"Error in conversation: {e}"
    
    def setup_microphone(
        self,
        device_name_prefix=None,
        auto_select=True
    ):
        try:
            input_devices = [(i, d) for i, d in enumerate(sd.query_devices()) if d['max_input_channels'] > 0]
            
            if not input_devices:
                print("No input devices detected!")
                return False
            
            if device_name_prefix:
                for idx, device in input_devices:
                    if device['name'].startswith(device_name_prefix):
                        self.mic_device_index = idx
                        self.mic_device = device
                        print(f"Selected microphone: {device['name']}")
                        return True
                
                print(f"No device found with prefix '{device_name_prefix}'")
            
            if auto_select:
                self.mic_device_index = sd.default.device[0]
                self.mic_device = sd.query_devices()[self.mic_device_index]
                print(f"Using default microphone: {self.mic_device['name']}")
                return True
            
            return False
            
        except Exception as e:
            print(f"Error setting up microphone: {e}")
            return False
    
    def list_microphones(self):
        try:
            return [
                {
                    'index': i,
                    'name': d['name'],
                    'sample_rate': d['default_samplerate'],
                    'channels': d['max_input_channels']
                }
                for i, d in enumerate(sd.query_devices())
                if d['max_input_channels'] > 0
            ]
        except Exception as e:
            print(f"Error listing microphones: {e}")
            return []
    
    def listen_continuous(
        self,
        frame_duration=2,
        language="en-US"
    ):
        if self.mic_device_index is None:
            if not self.setup_microphone(device_name_prefix="PCM", auto_select=True):
                return None
        
        try:
            sample_rate = int(self.mic_device['default_samplerate'])
            
            audio_data = sd.rec(
                int(frame_duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype='int16',
                device=self.mic_device_index
            )
            sd.wait()  
            audio_max = np.abs(audio_data).max()
            if audio_max < 100: 
                print(f"Warning: Audio level very low ({audio_max}). Check microphone volume!")
            
            # Save audio to temporary WAV file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_filename = temp_file.name
            
            with wave.open(temp_filename, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2) 
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data.tobytes())
            
            # Transcribe using GPT-4o
            text = self._transcribe_with_gpt4o(temp_filename, language)
            
            # Clean up temp file
            try:
                os.remove(temp_filename)
            except:
                pass
            
            return text
        
        except Exception as e:
            print(f"Recognition error: {e}")
            return None
    
    def listen_with_silence_detection(
        self,
        max_duration=30,
        silence_threshold=None,
        silence_duration=1.5,
        language="en-US",
        auto_calibrate=False
    ):
        if self.mic_device_index is None:
            print("Setting up microphone...")
            if not self.setup_microphone(device_name_prefix="PCM", auto_select=True):
                print("Failed to setup microphone")
                return None
        
        try:
            sample_rate = int(self.mic_device['default_samplerate'])
            if silence_threshold is None:
                silence_threshold = self.silence_threshold
            
            if auto_calibrate:
                print("Calibrating noise floor... (stay quiet for 2 seconds)")
                calibration_samples = []
                for _ in range(4):
                    chunk_data = sd.rec(
                        int(0.5 * sample_rate),
                        samplerate=sample_rate,
                        channels=1,
                        dtype='int16',
                        device=self.mic_device_index
                    )
                    sd.wait()
                    energy = np.abs(chunk_data).mean()
                    calibration_samples.append(energy)
                
                avg_noise = np.mean(calibration_samples)
                silence_threshold = avg_noise * 2.5
                self.silence_threshold = silence_threshold
                print(f"Noise floor: {avg_noise:.0f}, Speech threshold: {silence_threshold:.0f}")
            
            print(f"Listening... Speak now! (stops automatically when you finish)")
            
            stop_animation = threading.Event()
            anim_thread = threading.Thread(
                target=oled_listening_animation,
                args=(stop_animation,),
                daemon=True
            )
            anim_thread.start()
            
            all_audio_chunks = []
            silence_time = 0
            has_speech = False
            total_time = 0
            
            while total_time < max_duration:
                chunk_data = sd.rec(
                    int(0.5 * sample_rate),
                    samplerate=sample_rate,
                    channels=1,
                    dtype='int16',
                    device=self.mic_device_index
                )
                sd.wait()  
                
                all_audio_chunks.append(chunk_data.copy())
                total_time += 0.5
                
                energy = np.abs(chunk_data).mean()
                
                if energy > silence_threshold:
                    silence_time = 0
                    has_speech = True
                    print(".", end="", flush=True) 
                else:
                    if has_speech:
                        silence_time += 0.5
                        print("_", end="", flush=True) 
                if has_speech and silence_time >= silence_duration:
                    print("\nSilence detected, stopping...")
                    break
            
            if not has_speech:
                print("\nNo speech detected")
                stop_animation.set()
                anim_thread.join(timeout=0.1)
                return None
            
            print("\nProcessing speech")
            
            stop_animation.set()
            anim_thread.join(timeout=0.1)
            
            audio_data = np.concatenate(all_audio_chunks, axis=0)
            
            # Save audio to temporary WAV file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_filename = temp_file.name
            
            with wave.open(temp_filename, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data.tobytes())
            
            # Transcribe using GPT-4o
            text = self._transcribe_with_gpt4o(temp_filename, language)
            
            # Clean up temp file
            try:
                os.remove(temp_filename)
            except:
                pass
            
            if text:
                print(f"You said: {text}")
                return text
            else:
                print("Could not transcribe the audio.")
                return None
        
        except Exception as e:
            print(f"Error during speech recognition: {e}")
            import traceback
            traceback.print_exc()
            stop_animation.set()
            anim_thread.join(timeout=0.1)
            return None
    
    def listen(
        self,
        duration=5,
        language="en-US",
        show_all=False
    ):
        if self.mic_device_index is None:
            print("Setting up microphone...")
            if not self.setup_microphone(device_name_prefix="PCM", auto_select=True):
                print("Failed to setup microphone")
                return None
        
        try:
            sample_rate = int(self.mic_device['default_samplerate'])
            
            print(f"Recording for {duration} seconds... Speak now!")
            
            audio_data = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype='int16',
                device=self.mic_device_index
            )
            sd.wait() 
            
            print("Processing speech...")
            
            # Save audio to temporary WAV file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_filename = temp_file.name
            
            with wave.open(temp_filename, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data.tobytes())
            
            # Transcribe using GPT-4o
            text = self._transcribe_with_gpt4o(temp_filename, language)
            
            # Clean up temp file
            try:
                os.remove(temp_filename)
            except:
                pass
            
            if text:
                print(f"You said: {text}")
                return text
            else:
                print("Could not transcribe the audio.")
                return None
        
        except Exception as e:
            print(f"Error during speech recognition: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def speak(
    self, 
    text, 
    voice=None, 
    save_to=None
    ):
        try:
            anim_thread = threading.Thread(
                target=oled_speaking_animation,
                args=(max(2.0, len(text) * 0.15),),
                daemon=True
            )
            anim_thread.start()

            response = self.client.audio.speech.create(
                model=self.tts_model,
                voice=voice or self.tts_voice,
                input=text
            )

            if save_to:
                audio_path = save_to
                with open(audio_path, 'wb') as audio_file:
                    audio_file.write(response.content)
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
                    temp_audio.write(response.content)
                    audio_path = temp_audio.name

            for cmd in [
                ["mpv", "--really-quiet", audio_path],
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", audio_path],
                ["paplay", audio_path],
            ]:
                try:
                    subprocess.run(cmd, check=True, capture_output=True)
                    anim_thread.join(timeout=0.1)
                    if not save_to:
                        os.remove(audio_path)
                    return True
                except:
                    pass

            anim_thread.join(timeout=0.1)
            if not save_to:
                os.remove(audio_path)
            return False

        except Exception as e:
            print(f"Error speaking: {e}")
            return False

    def listen_and_ask(
        self,
        duration=5,
        language="en-US",
        use_rag=True,
        n_results=5,
        auto_stop=True,
        max_duration=30,
        silence_duration=1.5,
        speak_response=False,
        save_audio_to=None
    ):
        if auto_stop:
            query = self.listen_with_silence_detection(
                max_duration=max_duration,
                silence_duration=silence_duration,
                language=language
            )
        else:
            query = self.listen(
                duration=duration,
                language=language
            )
        
        if query is None:
            return None
        
        print("JARVIS is thinking")
        response = self.ask(query, use_rag=use_rag, n_results=n_results)
        
        if speak_response and response:
            self.speak(response, save_to=save_audio_to)
        
        return response
    
    def clear_context(self):
        self.context_chunks = []
        self.large_context = ""
        
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
        except:
            pass
        
        self.collection_name = f"jarvis_memory_{uuid.uuid4().hex[:8]}"
        self.collection = self.chroma_client.create_collection(
            name=self.collection_name
        )
        
        print("🧹 Context cleared. New collection created.")
    
    def get_stats(self):
        return {
            "total_chunks": len(self.context_chunks),
            "collection_name": self.collection_name,
            "model": self.model,
            "embedding_model": self.embedding_model,
            "context_size_chars": len(self.large_context)
        }

def create_jarvis(**kwargs):
    return JARVIS(**kwargs)
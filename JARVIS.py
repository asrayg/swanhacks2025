"""
JARVIS - A modular AI assistant library with RAG capabilities
Can be imported and used in any Python project
"""

import os
import pathlib
import uuid
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
import chromadb
from chromadb.config import Settings
import PyPDF2
import speech_recognition as sr
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
        api_key: Optional[str] = None,
        model: str = "gpt-5-nano",
        embedding_model: str = "text-embedding-3-small",
        db_path: Optional[str] = None,
        disable_telemetry: bool = True,
        tts_voice: str = "alloy",
        tts_model: str = "tts-1-hd"
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
        
        self.recognizer = sr.Recognizer()
        self.mic_device_index = None
        self.mic_device = None
        self.silence_threshold = 3000  
        self.tts_voice = tts_voice
        self.tts_model = tts_model 
        
        self.system_instructions = self._load_instructions()
        
        print(f"JARVIS initialized with collection: {self.collection_name}")
    
    def _load_instructions(self) -> str:
        instructions_file = pathlib.Path("jarvis_instructions.txt")
        
        if instructions_file.exists():
            try:
                instructions = instructions_file.read_text(encoding="utf-8")
                print("Loaded system instructions")
                return instructions
            except Exception as e:
                print(f"Could not load instructions: {e}")
        
        return """You are JARVIS, an intelligent AI assistant. 
Be helpful, accurate, and concise. Use provided context when available."""
    
    def set_silence_threshold(self, threshold: float) -> None:
        self.silence_threshold = threshold
        print(f"JARVIS silence threshold set to: {threshold:.0f}")
    
    def reload_instructions(self, instructions_path: Optional[str] = None) -> bool:
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
    
    def _embed_documents(self, texts: List[str]) -> List[List[float]]:
        res = self.client.embeddings.create(
            model=self.embedding_model,
            input=texts
        )
        return [d.embedding for d in res.data]
    
    def _embed_query(self, text: str) -> List[float]:
        res = self.client.embeddings.create(
            model=self.embedding_model,
            input=[text]
        )
        return res.data[0].embedding
    
    def _extract_text_from_file(self, filepath: pathlib.Path) -> str:
        text = ""
        
        if filepath.suffix.lower() == ".txt":
            text = filepath.read_text(encoding="utf-8", errors="ignore")
        elif filepath.suffix.lower() == ".pdf":
            try:
                reader = PyPDF2.PdfReader(str(filepath))
                text = "\n".join(
                    page.extract_text() or "" for page in reader.pages
                )
            except Exception as e:
                print(f"⚠️ Could not read {filepath.name}: {e}")
        
        return text.strip()
    
    def add_context_from_file(
        self,
        filepath: str,
        chunk_size: int = 2000,
        chunk_overlap: int = 200
    ) -> int:
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
        directory: str,
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
        recursive: bool = False
    ) -> int:
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
        text: str,
        chunk_size: int = 2000,
        chunk_overlap: int = 200
    ) -> int:
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
        query: str,
        n_results: int = 5
    ) -> List[str]:
        try:
            qvec = self._embed_query(query)
            results = self.collection.query(
                query_embeddings=[qvec],
                n_results=n_results
            )
            docs = results.get("documents", [[]])[0]
            return docs
        except Exception as e:
            print(f"Error retrieving context: {e}")
            return []
    
    def ask(
        self,
        query: str,
        use_rag: bool = True,
        n_results: int = 5,
        include_sources: bool = False,
        system_prompt: Optional[str] = None
    ) -> str:
        context_parts = []
        sources = []
        
        if use_rag and self.context_chunks:
            relevant_docs = self._retrieve_relevant_context(query, n_results)
            if relevant_docs:
                context_parts.append("Relevant information:")
                context_parts.extend(relevant_docs)
                sources = ["Vector Database"] * len(relevant_docs)
        
        if system_prompt is None:
            system_prompt = self.system_instructions
        
        messages = [{"role": "system", "content": system_prompt}]
        
        if context_parts:
            context_message = "\n\n".join(context_parts)
            messages.append({
                "role": "user",
                "content": f"Context:\n{context_message}\n\nQuestion: {query}"
            })
        else:
            messages.append({"role": "user", "content": query})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            answer = response.choices[0].message.content
            
            if include_sources and sources:
                answer += f"\n\n[Sources: {', '.join(set(sources))}]"
            
            return answer
        
        except Exception as e:
            return f"Error generating response: {e}"
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        use_context: bool = True,
        n_results: int = 3
    ) -> str:
        if not messages:
            return "No messages provided."
        
        if use_context and self.context_chunks:
            last_message = messages[-1]["content"]
            relevant_docs = self._retrieve_relevant_context(
                last_message,
                n_results
            )
            
            if relevant_docs:
                context_msg = "Relevant context:\n" + "\n\n".join(relevant_docs)
                messages.insert(-1, {"role": "system", "content": context_msg})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            return response.choices[0].message.content
        
        except Exception as e:
            return f"Error in conversation: {e}"
    
    def setup_microphone(
        self,
        device_name_prefix: Optional[str] = None,
        auto_select: bool = True
    ) -> bool:
        try:
            devices = sd.query_devices()
            input_devices = [(i, d) for i, d in enumerate(devices) if d['max_input_channels'] > 0]
            
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
                self.mic_device = devices[self.mic_device_index]
                print(f"Using default microphone: {self.mic_device['name']}")
                return True
            
            return False
            
        except Exception as e:
            print(f"Error setting up microphone: {e}")
            return False
    
    def list_microphones(self) -> List[Dict[str, Any]]:
        try:
            devices = sd.query_devices()
            input_devices = []
            
            for i, d in enumerate(devices):
                if d['max_input_channels'] > 0:
                    input_devices.append({
                        'index': i,
                        'name': d['name'],
                        'sample_rate': d['default_samplerate'],
                        'channels': d['max_input_channels']
                    })
            
            return input_devices
            
        except Exception as e:
            print(f"Error listing microphones: {e}")
            return []
    
    def listen_continuous(
        self,
        frame_duration: int = 2,
        language: str = "en-US"
    ) -> Optional[str]:
        if self.mic_device_index is None:
            if not self.setup_microphone(device_name_prefix="PCM", auto_select=True):
                return None
        
        try:
            sample_rate = int(self.mic_device['default_samplerate'])
            channels = 1
            
            audio_data = sd.rec(
                int(frame_duration * sample_rate),
                samplerate=sample_rate,
                channels=channels,
                dtype='int16',
                device=self.mic_device_index
            )
            sd.wait()  
            audio_max = np.abs(audio_data).max()
            if audio_max < 100: 
                print(f"Warning: Audio level very low ({audio_max}). Check microphone volume!")
            
            wav_io = io.BytesIO()
            with wave.open(wav_io, 'wb') as wav_file:
                wav_file.setnchannels(channels)
                wav_file.setsampwidth(2) 
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data.tobytes())
            
            wav_io.seek(0)
            
            with sr.AudioFile(wav_io) as source:
                audio = self.recognizer.record(source)
            
            text = self.recognizer.recognize_google(
                audio,
                language=language
            )
            
            return text
        
        except sr.UnknownValueError:
            return None
        
        except sr.RequestError as e:
            print(f"Google API request failed: {e}")
            return None
        
        except Exception as e:
            print(f"Recognition error: {e}")
            return None
    
    def listen_with_silence_detection(
        self,
        max_duration: int = 30,
        silence_threshold: Optional[int] = None,
        silence_duration: float = 1.5,
        language: str = "en-US",
        auto_calibrate: bool = False
    ) -> Optional[str]:
        if self.mic_device_index is None:
            print("Setting up microphone...")
            if not self.setup_microphone(device_name_prefix="PCM", auto_select=True):
                print("Failed to setup microphone")
                return None
        
        try:
            sample_rate = int(self.mic_device['default_samplerate'])
            channels = 1
            chunk_duration = 0.5  
            if silence_threshold is None:
                silence_threshold = self.silence_threshold
            
            if auto_calibrate:
                print("Calibrating noise floor... (stay quiet for 2 seconds)")
                calibration_samples = []
                for _ in range(4):
                    chunk_data = sd.rec(
                        int(chunk_duration * sample_rate),
                        samplerate=sample_rate,
                        channels=channels,
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
                    int(chunk_duration * sample_rate),
                    samplerate=sample_rate,
                    channels=channels,
                    dtype='int16',
                    device=self.mic_device_index
                )
                sd.wait()  
                
                all_audio_chunks.append(chunk_data.copy())
                total_time += chunk_duration
                
                energy = np.abs(chunk_data).mean()
                
                if energy > silence_threshold:
                    silence_time = 0
                    has_speech = True
                    print(".", end="", flush=True) 
                else:
                    if has_speech:
                        silence_time += chunk_duration
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
            
            wav_io = io.BytesIO()
            with wave.open(wav_io, 'wb') as wav_file:
                wav_file.setnchannels(channels)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data.tobytes())
            
            wav_io.seek(0)
            
            with sr.AudioFile(wav_io) as source:
                audio = self.recognizer.record(source)
            
            text = self.recognizer.recognize_google(
                audio,
                language=language
            )
            
            print(f"You said: {text}")
            return text
        
        except sr.UnknownValueError:
            print("Could not understand the audio.")
            stop_animation.set()
            anim_thread.join(timeout=0.1)
            return None
        
        except sr.RequestError as e:
            print(f"Speech recognition service error: {e}")
            stop_animation.set()
            anim_thread.join(timeout=0.1)
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
        duration: int = 5,
        language: str = "en-US",
        show_all: bool = False
    ) -> Optional[str]:
        if self.mic_device_index is None:
            print("Setting up microphone...")
            if not self.setup_microphone(device_name_prefix="PCM", auto_select=True):
                print("Failed to setup microphone")
                return None
        
        try:
            sample_rate = int(self.mic_device['default_samplerate'])
            channels = 1
            
            print(f"Recording for {duration} seconds... Speak now!")
            
            audio_data = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=channels,
                dtype='int16',
                device=self.mic_device_index
            )
            sd.wait() 
            
            print("Processing speech...")
            
            wav_io = io.BytesIO()
            with wave.open(wav_io, 'wb') as wav_file:
                wav_file.setnchannels(channels)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data.tobytes())
            
            wav_io.seek(0)
            
            with sr.AudioFile(wav_io) as source:
                audio = self.recognizer.record(source)
            
            text = self.recognizer.recognize_google(
                audio,
                language=language,
                show_all=show_all
            )
            
            print(f"You said: {text}")
            return text
        
        except sr.UnknownValueError:
            print("Could not understand the audio.")
            return None
        
        except sr.RequestError as e:
            print(f"Speech recognition service error: {e}")
            return None
        
        except Exception as e:
            print(f"Error during speech recognition: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def speak(
    self, 
    text: str, 
    voice: Optional[str] = None, 
    save_to: Optional[str] = None
    ) -> bool:
        try:
            voice = voice or self.tts_voice

            anim_thread = threading.Thread(
                target=oled_speaking_animation,
                args=(max(2.0, len(text) * 0.15),),
                daemon=True
            )
            anim_thread.start()

            response = self.client.audio.speech.create(
                model=self.tts_model,
                voice=voice,
                input=text
            )

            if save_to:
                audio_path = save_to
                with open(audio_path, 'wb') as audio_file:
                    audio_file.write(response.content)
                should_delete = False
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
                    temp_audio.write(response.content)
                    audio_path = temp_audio.name
                should_delete = True

            played = False
            for cmd in [
                ["mpv", "--really-quiet", audio_path],
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", audio_path],
                ["paplay", audio_path],
            ]:
                try:
                    subprocess.run(cmd, check=True, capture_output=True)
                    played = True
                    break
                except:
                    pass

            anim_thread.join(timeout=0.1)

            if should_delete:
                os.remove(audio_path)

            return played

        except Exception as e:
            print(f"Error speaking: {e}")
            return False

    def listen_and_ask(
        self,
        duration: int = 5,
        language: str = "en-US",
        use_rag: bool = True,
        n_results: int = 5,
        auto_stop: bool = True,
        max_duration: int = 30,
        silence_duration: float = 1.5,
        speak_response: bool = False,
        save_audio_to: Optional[str] = None
    ) -> Optional[str]:
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
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_chunks": len(self.context_chunks),
            "collection_name": self.collection_name,
            "model": self.model,
            "embedding_model": self.embedding_model,
            "context_size_chars": len(self.large_context)
        }

def create_jarvis(**kwargs) -> JARVIS:
    return JARVIS(**kwargs)
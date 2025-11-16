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


class JARVIS:
    """
    JARVIS AI Assistant Library
    
    A modular chatbot that supports:
    - Large text file context windows
    - RAG (Retrieval-Augmented Generation) with ChromaDB
    - Document ingestion from files and directories
    - Easy integration into other projects
    
    Usage:
        jarvis = JARVIS()
        jarvis.add_context_from_file("large_document.txt")
        response = jarvis.ask("What is this document about?")
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-5.1",
        embedding_model: str = "text-embedding-3-small",
        temperature: float = 0.2,
        db_path: Optional[str] = None,
        disable_telemetry: bool = True,
        tts_voice: str = "alloy",
        tts_model: str = "tts-1-hd"
    ):
        """
        Initialize JARVIS assistant.
        
        Args:
            api_key: OpenAI API key (if None, loads from environment)
            model: OpenAI chat model to use
            embedding_model: OpenAI embedding model to use
            temperature: Response temperature (0.0-1.0)
            db_path: Path to ChromaDB storage (default: ./jarvis_db)
            disable_telemetry: Disable ChromaDB telemetry
            tts_voice: OpenAI TTS voice (alloy, echo, fable, onyx, nova, shimmer)
            tts_model: OpenAI TTS model (tts-1 or tts-1-hd)
        """
        # Disable telemetry if requested
        if disable_telemetry:
            os.environ["ANONYMIZED_TELEMETRY"] = "False"
            os.environ["CHROMA_TELEMETRY_ENABLED"] = "False"
            os.environ["CHROMA_USE_OPENAI_EMBEDDINGS"] = "False"
            os.environ["CHROMA_DISABLE_TELEMETRY"] = "True"
        
        # Load environment variables
        load_dotenv(find_dotenv())
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise RuntimeError(
                "OpenAI API key not found. Please provide via api_key parameter "
                "or set OPENAI_API_KEY environment variable."
            )
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.embedding_model = embedding_model
        self.temperature = temperature
        
        # Setup database
        self.db_path = pathlib.Path(db_path or "./jarvis_db")
        self.db_path.mkdir(exist_ok=True)
        
        # Initialize ChromaDB
        settings = Settings(
            persist_directory=str(self.db_path),
            anonymized_telemetry=False,
            is_persistent=True
        )
        self.chroma_client = chromadb.Client(settings)
        
        # Create or get collection
        self.collection_name = f"jarvis_memory_{uuid.uuid4().hex[:8]}"
        self.collection = self.chroma_client.create_collection(
            name=self.collection_name
        )
        
        # Context management
        self.context_chunks = []
        self.large_context = ""
        
        # Speech recognition
        self.recognizer = sr.Recognizer()
        self.mic_device_index = None
        self.mic_device = None
        
        # Text-to-speech settings
        self.tts_voice = tts_voice  # Options: alloy, echo, fable, onyx, nova, shimmer
        self.tts_model = tts_model  # tts-1 or tts-1-hd for higher quality
        
        # Load system instructions
        self.system_instructions = self._load_instructions()
        
        print(f"🤖 JARVIS initialized with collection: {self.collection_name}")
    
    def _load_instructions(self) -> str:
        """Load system instructions from file."""
        instructions_file = pathlib.Path("jarvis_instructions.txt")
        
        if instructions_file.exists():
            try:
                instructions = instructions_file.read_text(encoding="utf-8")
                print("📋 Loaded system instructions")
                return instructions
            except Exception as e:
                print(f"⚠️ Could not load instructions: {e}")
        
        # Default instructions if file doesn't exist
        return """You are JARVIS, an intelligent AI assistant. 
Be helpful, accurate, and concise. Use provided context when available."""
    
    def reload_instructions(self, instructions_path: Optional[str] = None) -> bool:
        """
        Reload system instructions from file.
        
        Args:
            instructions_path: Optional path to instructions file (default: jarvis_instructions.txt)
        
        Returns:
            True if successfully reloaded
        """
        if instructions_path:
            instructions_file = pathlib.Path(instructions_path)
        else:
            instructions_file = pathlib.Path("jarvis_instructions.txt")
        
        if not instructions_file.exists():
            print(f"❌ Instructions file not found: {instructions_file}")
            return False
        
        try:
            self.system_instructions = instructions_file.read_text(encoding="utf-8")
            print(f"✅ Reloaded instructions from {instructions_file}")
            return True
        except Exception as e:
            print(f"❌ Error reloading instructions: {e}")
            return False
    
    def _embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple documents."""
        res = self.client.embeddings.create(
            model=self.embedding_model,
            input=texts
        )
        return [d.embedding for d in res.data]
    
    def _embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query."""
        res = self.client.embeddings.create(
            model=self.embedding_model,
            input=[text]
        )
        return res.data[0].embedding
    
    def _extract_text_from_file(self, filepath: pathlib.Path) -> str:
        """Extract text from .txt or .pdf file."""
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
        """
        Add a large text file to JARVIS's context window.
        
        Args:
            filepath: Path to .txt or .pdf file
            chunk_size: Size of text chunks for vector storage
            chunk_overlap: Overlap between chunks for better context
        
        Returns:
            Number of chunks added
        """
        filepath = pathlib.Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        text = self._extract_text_from_file(filepath)
        
        if not text:
            print(f"⚠️ No text extracted from {filepath.name}")
            return 0
        
        # Store full text for direct context
        self.large_context += "\n\n" + text
        
        # Split into overlapping chunks
        chunks = []
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunk = text[i:i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        
        # Generate embeddings and store in vector DB
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
            print(f"✅ Added {len(chunks)} chunks from {filepath.name}")
        
        return len(chunks)
    
    def add_context_from_directory(
        self,
        directory: str,
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
        recursive: bool = False
    ) -> int:
        """
        Add all .txt and .pdf files from a directory to context.
        
        Args:
            directory: Path to directory containing documents
            chunk_size: Size of text chunks for vector storage
            chunk_overlap: Overlap between chunks
            recursive: If True, search subdirectories recursively
        
        Returns:
            Total number of chunks added
        """
        directory = pathlib.Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        total_chunks = 0
        file_count = 0
        
        # Choose glob pattern based on recursive flag
        if recursive:
            pattern = "**/*"
            print(f"🔍 Recursively searching {directory} for documents...")
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
                    print(f"⚠️ Error processing {file.name}: {e}")
        
        print(f"📚 Total: {total_chunks} chunks from {file_count} files")
        return total_chunks
    
    def add_context_from_text(
        self,
        text: str,
        chunk_size: int = 2000,
        chunk_overlap: int = 200
    ) -> int:
        """
        Add raw text to JARVIS's context window.
        
        Args:
            text: Text string to add
            chunk_size: Size of text chunks for vector storage
            chunk_overlap: Overlap between chunks
        
        Returns:
            Number of chunks added
        """
        if not text.strip():
            return 0
        
        # Store full text for direct context
        self.large_context += "\n\n" + text
        
        # Split into overlapping chunks
        chunks = []
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunk = text[i:i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        
        # Generate embeddings and store in vector DB
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
            print(f"✅ Added {len(chunks)} chunks from text")
        
        return len(chunks)
    
    def _retrieve_relevant_context(
        self,
        query: str,
        n_results: int = 5
    ) -> List[str]:
        """Retrieve relevant context chunks for a query."""
        try:
            qvec = self._embed_query(query)
            results = self.collection.query(
                query_embeddings=[qvec],
                n_results=n_results
            )
            docs = results.get("documents", [[]])[0]
            return docs
        except Exception as e:
            print(f"⚠️ Error retrieving context: {e}")
            return []
    
    def ask(
        self,
        query: str,
        use_rag: bool = True,
        n_results: int = 5,
        include_sources: bool = False,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Ask JARVIS a question.
        
        Args:
            query: The question to ask
            use_rag: Use RAG to retrieve relevant context
            n_results: Number of context chunks to retrieve
            include_sources: Include source information in response
            system_prompt: Custom system prompt (overrides default)
        
        Returns:
            JARVIS's response
        """
        # Build context
        context_parts = []
        sources = []
        
        if use_rag and self.context_chunks:
            relevant_docs = self._retrieve_relevant_context(query, n_results)
            if relevant_docs:
                context_parts.append("Relevant information:")
                context_parts.extend(relevant_docs)
                sources = ["Vector Database"] * len(relevant_docs)
        
        # Build prompt
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
        
        # Get response from OpenAI
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature
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
        """
        Multi-turn conversation with JARVIS.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            use_context: Inject relevant context based on last message
            n_results: Number of context chunks to retrieve
        
        Returns:
            JARVIS's response
        """
        if not messages:
            return "No messages provided."
        
        # Inject context if requested
        if use_context and self.context_chunks:
            last_message = messages[-1]["content"]
            relevant_docs = self._retrieve_relevant_context(
                last_message,
                n_results
            )
            
            if relevant_docs:
                context_msg = "Relevant context:\n" + "\n\n".join(relevant_docs)
                messages.insert(-1, {"role": "system", "content": context_msg})
        
        # Get response
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature
            )
            return response.choices[0].message.content
        
        except Exception as e:
            return f"Error in conversation: {e}"
    
    def setup_microphone(
        self,
        device_name_prefix: Optional[str] = None,
        auto_select: bool = True
    ) -> bool:
        """
        Setup microphone device for speech recognition.
        
        Args:
            device_name_prefix: Prefix to match (e.g., "PCM", "NoiseTorch")
            auto_select: Automatically select first matching device
        
        Returns:
            True if device was set up successfully
        """
        try:
            devices = sd.query_devices()
            input_devices = [(i, d) for i, d in enumerate(devices) if d['max_input_channels'] > 0]
            
            if not input_devices:
                print("⚠️ No input devices detected!")
                return False
            
            # Find device by prefix if specified
            if device_name_prefix:
                for idx, device in input_devices:
                    if device['name'].startswith(device_name_prefix):
                        self.mic_device_index = idx
                        self.mic_device = device
                        print(f"✅ Selected microphone: {device['name']}")
                        return True
                
                print(f"⚠️ No device found with prefix '{device_name_prefix}'")
            
            # Use default device
            if auto_select:
                self.mic_device_index = sd.default.device[0]  # Input device
                self.mic_device = devices[self.mic_device_index]
                print(f"✅ Using default microphone: {self.mic_device['name']}")
                return True
            
            return False
            
        except Exception as e:
            print(f"❌ Error setting up microphone: {e}")
            return False
    
    def list_microphones(self) -> List[Dict[str, Any]]:
        """
        List all available input devices.
        
        Returns:
            List of device info dictionaries
        """
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
            print(f"❌ Error listing microphones: {e}")
            return []
    
    def listen_continuous(
        self,
        frame_duration: int = 2,
        language: str = "en-US"
    ) -> Optional[str]:
        """
        Listen with overlapping frames for better wake word detection.
        Records frames with 50% overlap (e.g., 2s frames with 1s stride).
        
        Args:
            frame_duration: Duration of each recording frame in seconds
            language: Language code (e.g., 'en-US', 'es-ES')
        
        Returns:
            Transcribed text or None if recognition failed
        """
        # Setup microphone if not already done
        if self.mic_device_index is None:
            if not self.setup_microphone(device_name_prefix="PCM", auto_select=True):
                return None
        
        try:
            sample_rate = int(self.mic_device['default_samplerate'])
            channels = 1
            
            # Record audio using sounddevice
            audio_data = sd.rec(
                int(frame_duration * sample_rate),
                samplerate=sample_rate,
                channels=channels,
                dtype='int16',
                device=self.mic_device_index
            )
            sd.wait()  # Wait until recording is finished
            
            # Debug: Check audio levels
            audio_max = np.abs(audio_data).max()
            if audio_max < 100:  # Very quiet audio
                print(f"⚠️ Warning: Audio level very low ({audio_max}). Check microphone volume!")
            
            # Convert to WAV format in memory
            wav_io = io.BytesIO()
            with wave.open(wav_io, 'wb') as wav_file:
                wav_file.setnchannels(channels)
                wav_file.setsampwidth(2)  # 2 bytes for int16
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data.tobytes())
            
            # Reset to beginning of BytesIO
            wav_io.seek(0)
            
            # Use speech_recognition to transcribe
            with sr.AudioFile(wav_io) as source:
                audio = self.recognizer.record(source)
            
            # Recognize speech using Google Speech Recognition
            text = self.recognizer.recognize_google(
                audio,
                language=language
            )
            
            return text
        
        except sr.UnknownValueError:
            # Google couldn't understand the audio
            # This usually means audio is too quiet, corrupted, or just silence/noise
            return None
        
        except sr.RequestError as e:
            print(f"❌ Google API request failed: {e}")
            return None
        
        except Exception as e:
            print(f"❌ Recognition error: {e}")
            return None
    
    def listen_with_silence_detection(
        self,
        max_duration: int = 30,
        silence_threshold: int = 1500,
        silence_duration: float = 1.5,
        language: str = "en-US"
    ) -> Optional[str]:
        """
        Listen to microphone and automatically stop when user stops talking.
        Uses chunk-based recording to avoid streaming issues with virtual devices.
        
        Args:
            max_duration: Maximum recording duration in seconds
            silence_threshold: Energy threshold below which is considered silence
            silence_duration: Seconds of silence before stopping
            language: Language code (e.g., 'en-US', 'es-ES')
        
        Returns:
            Transcribed text or None if recognition failed
        """
        # Setup microphone if not already done
        if self.mic_device_index is None:
            print("🔧 Setting up microphone...")
            if not self.setup_microphone(device_name_prefix="PCM", auto_select=True):
                print("❌ Failed to setup microphone")
                return None
        
        try:
            sample_rate = int(self.mic_device['default_samplerate'])
            channels = 1
            chunk_duration = 0.5  # 500ms chunks (safer for virtual devices)
            
            print(f"🎤 Listening... Speak now! (stops automatically when you finish)")
            
            # Storage for audio data
            all_audio_chunks = []
            silence_time = 0
            has_speech = False
            total_time = 0
            
            while total_time < max_duration:
                # Record a chunk
                chunk_data = sd.rec(
                    int(chunk_duration * sample_rate),
                    samplerate=sample_rate,
                    channels=channels,
                    dtype='int16',
                    device=self.mic_device_index
                )
                sd.wait()  # Wait for chunk to complete
                
                all_audio_chunks.append(chunk_data.copy())
                total_time += chunk_duration
                
                # Calculate energy (volume) of chunk
                energy = np.abs(chunk_data).mean()
                
                if energy > silence_threshold:
                    # Speech detected
                    silence_time = 0
                    has_speech = True
                    print(".", end="", flush=True)  # Visual feedback
                else:
                    # Silence detected
                    if has_speech:
                        silence_time += chunk_duration
                        print("_", end="", flush=True)  # Visual feedback
                
                # Stop if we've had enough silence after speech
                if has_speech and silence_time >= silence_duration:
                    print("\n🔇 Silence detected, stopping...")
                    break
            
            if not has_speech:
                print("\n⚠️ No speech detected")
                return None
            
            print("\n🔄 Processing speech...")
            
            # Concatenate all audio chunks
            audio_data = np.concatenate(all_audio_chunks, axis=0)
            
            # Convert to WAV format in memory
            wav_io = io.BytesIO()
            with wave.open(wav_io, 'wb') as wav_file:
                wav_file.setnchannels(channels)
                wav_file.setsampwidth(2)  # 2 bytes for int16
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data.tobytes())
            
            # Reset to beginning of BytesIO
            wav_io.seek(0)
            
            # Use speech_recognition to transcribe
            with sr.AudioFile(wav_io) as source:
                audio = self.recognizer.record(source)
            
            # Recognize speech using Google Speech Recognition
            text = self.recognizer.recognize_google(
                audio,
                language=language
            )
            
            print(f"📝 You said: {text}")
            return text
        
        except sr.UnknownValueError:
            print("❓ Could not understand the audio.")
            return None
        
        except sr.RequestError as e:
            print(f"❌ Speech recognition service error: {e}")
            return None
        
        except Exception as e:
            print(f"❌ Error during speech recognition: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def listen(
        self,
        duration: int = 5,
        language: str = "en-US",
        show_all: bool = False
    ) -> Optional[str]:
        """
        Listen to microphone and convert speech to text using sounddevice.
        
        Args:
            duration: Recording duration in seconds
            language: Language code (e.g., 'en-US', 'es-ES')
            show_all: Return all alternative transcriptions
        
        Returns:
            Transcribed text or None if recognition failed
        """
        # Setup microphone if not already done
        if self.mic_device_index is None:
            print("🔧 Setting up microphone...")
            if not self.setup_microphone(device_name_prefix="PCM", auto_select=True):
                print("❌ Failed to setup microphone")
                return None
        
        try:
            sample_rate = int(self.mic_device['default_samplerate'])
            channels = 1
            
            print(f"🎤 Recording for {duration} seconds... Speak now!")
            
            # Record audio using sounddevice
            audio_data = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=channels,
                dtype='int16',
                device=self.mic_device_index
            )
            sd.wait()  # Wait until recording is finished
            
            print("🔄 Processing speech...")
            
            # Convert to WAV format in memory
            wav_io = io.BytesIO()
            with wave.open(wav_io, 'wb') as wav_file:
                wav_file.setnchannels(channels)
                wav_file.setsampwidth(2)  # 2 bytes for int16
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data.tobytes())
            
            # Reset to beginning of BytesIO
            wav_io.seek(0)
            
            # Use speech_recognition to transcribe
            with sr.AudioFile(wav_io) as source:
                audio = self.recognizer.record(source)
            
            # Recognize speech using Google Speech Recognition
            text = self.recognizer.recognize_google(
                audio,
                language=language,
                show_all=show_all
            )
            
            print(f"📝 You said: {text}")
            return text
        
        except sr.UnknownValueError:
            print("❓ Could not understand the audio.")
            return None
        
        except sr.RequestError as e:
            print(f"❌ Speech recognition service error: {e}")
            return None
        
        except Exception as e:
            print(f"❌ Error during speech recognition: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def speak(
        self, 
        text: str, 
        voice: Optional[str] = None, 
        save_to: Optional[str] = None
    ) -> bool:
        """
        Speak text using OpenAI's TTS API and optionally save to file.
        
        Args:
            text: The text to speak
            voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer)
            save_to: Optional path to save the audio file (e.g., "output.mp3")
        
        Returns:
            True if successful, False otherwise
        """
        try:
            voice = voice or self.tts_voice
            
            # Generate speech using OpenAI TTS
            response = self.client.audio.speech.create(
                model=self.tts_model,
                voice=voice,
                input=text
            )
            
            # Determine file path
            if save_to:
                # Save to specified file
                audio_path = save_to
                with open(audio_path, 'wb') as audio_file:
                    audio_file.write(response.content)
                print(f"💾 Audio saved to: {audio_path}")
                should_delete = False
            else:
                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
                    temp_audio.write(response.content)
                    audio_path = temp_audio.name
                should_delete = True
            
            # Try to play using available audio player
            try:
                # Try mpv (common on Linux)
                subprocess.run(
                    ["mpv", "--really-quiet", audio_path],
                    check=True,
                    capture_output=True
                )
            except (FileNotFoundError, subprocess.CalledProcessError):
                try:
                    # Try ffplay (from ffmpeg)
                    subprocess.run(
                        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", audio_path],
                        check=True,
                        capture_output=True
                    )
                except (FileNotFoundError, subprocess.CalledProcessError):
                    # Try paplay (PulseAudio)
                    subprocess.run(
                        ["paplay", audio_path],
                        check=True,
                        capture_output=True
                    )
            
            # Clean up temp file if needed
            if should_delete:
                os.remove(audio_path)
            
            return True
            
        except Exception as e:
            print(f"❌ Error speaking: {e}")
            import traceback
            traceback.print_exc()
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
        """
        Listen to speech input and get JARVIS's response.
        
        Convenience method that combines listen() and ask().
        
        Args:
            duration: Recording duration in seconds (if auto_stop=False)
            language: Language code for speech recognition
            use_rag: Use RAG to retrieve relevant context
            n_results: Number of context chunks to retrieve
            auto_stop: Automatically stop when user stops talking
            max_duration: Maximum recording duration if auto_stop=True
            silence_duration: Seconds of silence before auto-stopping
            speak_response: Speak the response using TTS
            save_audio_to: Optional path to save audio response (e.g., "response.mp3")
        
        Returns:
            JARVIS's response or None if speech recognition failed
        """
        # Listen for speech with or without auto-stop
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
        
        # Get response from JARVIS
        print("🤖 JARVIS is thinking...")
        response = self.ask(query, use_rag=use_rag, n_results=n_results)
        
        # Speak response if requested
        if speak_response and response:
            self.speak(response, save_to=save_audio_to)
        
        return response
    
    def clear_context(self):
        """Clear all stored context and reset the database."""
        self.context_chunks = []
        self.large_context = ""
        
        # Create new collection
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
        """Get statistics about JARVIS's current state."""
        return {
            "total_chunks": len(self.context_chunks),
            "collection_name": self.collection_name,
            "model": self.model,
            "embedding_model": self.embedding_model,
            "context_size_chars": len(self.large_context)
        }


# Convenience function for quick usage
def create_jarvis(**kwargs) -> JARVIS:
    """
    Create a JARVIS instance with default settings.
    
    Usage:
        jarvis = create_jarvis()
        jarvis.add_context_from_file("document.txt")
        response = jarvis.ask("Tell me about this document")
    """
    return JARVIS(**kwargs)


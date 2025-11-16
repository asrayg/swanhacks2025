from JARVIS import create_jarvis
from translate import SpanishEnglishTranslator
import os
from datetime import datetime
import threading
import sys
import time
import detect
import sounddevice as sd
import numpy as np

oled_control_lock = threading.Lock()
oled_control_owner = None 


def calibrate_noise_floor(sample_rate=16000, channels=1):
    
    chunk_duration = 0.5  
    calibration_samples = []
    
    try:
        for _ in range(4): 
            chunk_data = sd.rec(
                int(chunk_duration * sample_rate),
                samplerate=sample_rate,
                channels=channels,
                dtype='int16'
            )
            sd.wait()
            energy = np.abs(chunk_data).mean()
            calibration_samples.append(energy)
        
        avg_noise = np.mean(calibration_samples)
        silence_threshold = avg_noise +100
        
        print(f"Noise floor: {avg_noise:.0f}")
        print(f"Speech threshold: {silence_threshold:.0f}\n")
        
        return silence_threshold
        
    except Exception as e:
        print(f"Error during calibration: {e}")
        print("Using default threshold of 10000\n")
        return 10000


def listen_for_wake_words(jarvis, frame_duration=2):
    text = jarvis.listen_continuous(frame_duration=frame_duration, language="en-US")
    
    if text:
        text_lower = text.lower()
        if "jeff" in text_lower:
            return "jeff"
        elif "translate" in text_lower:
            return "translate"
    
    return None

def main():
    print("="*60)
    print("JARVIS - Nursing Assistant")
    print("="*60)
    print("\nInitializing JARVIS...")
    print("(Loading system instructions from jarvis_instructions.txt)")
    jarvis = create_jarvis()
    
    print("\nSetting up microphone...")
    jarvis.setup_microphone(device_name_prefix="PCM", auto_select=True)
    
    silence_threshold = calibrate_noise_floor()
    
    jarvis.set_silence_threshold(silence_threshold)
    
    print("Loading clinical context...")
    try:
        if os.path.exists("output"):
            chunks = jarvis.add_context_from_directory("output", recursive=True)
            if chunks > 0:
                print(f"Loaded clinical reports from output folder")
        
        if os.path.exists("jarvis_kb.txt"):
            jarvis.add_context_from_file("jarvis_kb.txt")
            print(f"Loaded knowledge base")
    except Exception as e:
        print(f"Warning loading context: {e}")
    
    translator = SpanishEnglishTranslator(silence_threshold=silence_threshold)
    
    print("\n" + "="*60)
    print("JARVIS Nursing Assistant - Voice Interface")
    print("="*60)
    print("\n Wake Words:")
    print("   • Say 'JEFF' → Activate JARVIS nursing assistant")
    print("   • Say 'TRANSLATE' → Activate Spanish-English translator")
    print("\n Features:")
    print("   - Auto-stop when you finish speaking")
    print("   - Voice responses with OpenAI TTS")
    print("   - JARVIS: Clinical context from reports")
    print("   - Translator: Real-time Spanish ↔ English")
    print("   - Press Ctrl+C to exit")
    print("\n  Status: Ready - listening for wake words")
    print("="*60 + "\n")
    
    frame_count = 0
    try:
        while True:
            frame_count += 1
            print(f" Listening [Frame {frame_count}] - Say 'Jeff' or 'Translate'...")
            
            wake_word = listen_for_wake_words(jarvis, frame_duration=2)
            
            if wake_word == "jeff":
                print("Wake word 'Jeff' detected - JARVIS mode!")
                print("Listening for your question...\n")
                
                oled_control_lock.acquire()
                oled_control_owner = "JARVIS"
                print("OLED control: JARVIS (detect paused)")
                
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    audio_filename = f"jarvis_response_{timestamp}.mp3"
                    
                    response = jarvis.listen_and_ask(
                        auto_stop=True,        
                        max_duration=30,         
                        silence_duration=1.5,    
                        use_rag=True,
                        speak_response=True,
                        save_audio_to=audio_filename 
                    )
                    
                    if response:
                        print(f"\n JARVIS: {response}")
                        print(f"(Response spoken and saved to {audio_filename})\n")
                    else:
                        print("Could not process your question. Please try again.\n")
                finally:
                    oled_control_owner = None
                    oled_control_lock.release()
                    print("OLED control: Released (detect resumed)\n")
                
                print("-" * 60 + "\n")
                frame_count = 0  
            
            elif wake_word == "translate":
                print("Wake word 'Translate' detected - Translator mode!")
                print("Starting continuous translator...")
                print("Person speaks English → Translated to Spanish + spoken")
                print("Person speaks Spanish → Translated to English + spoken")
                print("\n   To exit, say any of:")
                print("   • 'stop translating' or 'end translating'")
                print("   • 'stop translation' or 'end translation'")
                print("   • Or press Ctrl+C\n")
                print("="*60 + "\n")
                
                # Acquire OLED control - detect will pause updates
                oled_control_lock.acquire()
                oled_control_owner = "TRANSLATOR"
                print("OLED control: TRANSLATOR (detect paused)\n")
                
                # Enter continuous translator mode
                translator.is_running = True
                conversation_count = 0
                
                try:
                    while translator.is_running:
                        conversation_count += 1
                        print(f"[Turn {conversation_count}] Listening... Speak now!")
                        
                        # Record with silence detection
                        audio_file = translator.record_with_silence_detection()
                        
                        if not audio_file:
                            continue
                        
                        text, detected_lang = translator.transcribe_audio(audio_file)
                        
                        if not text:
                            continue
                        
                        # Check for exit commands
                        text_lower = text.lower()
                        exit_phrases = [
                            'exit translator',
                            'stop translator', 
                            'stop translation',
                            'end translation',
                            'stop translating',
                            'end translating'
                            'stop',
                            'end',
                            'exit',
                            'quit',
                            'cancel',
                            'stop all',
                            'end all',
                            'exit all',
                            'quit all',
                            'translat'
                        ]
                        if any(phrase in text_lower for phrase in exit_phrases):
                            print("\nExit command detected. Leaving translator mode...\n")
                            break
                        
                        timestamp_str = datetime.now().strftime("%H:%M:%S")
                        
                        language = translator.detect_language(text, detected_lang)
                        
                        if language == "spanish":
                            print(f"\n{'='*60}")
                            print(f"[{timestamp_str}] SPANISH DETECTED:")
                            print(f"  {text}")
                            print(f"  (Whisper detected: {detected_lang})")
                            
                            english_translation = translator.translate_to_english(text)
                            
                            if english_translation:
                                print(f"\n[{timestamp_str}]  ENGLISH TRANSLATION:")
                                print(f"  {english_translation}")
                                print(f"\n Speaking in English...")
                                translator.speak_text(english_translation)
                                print("TTS completed")
                            print(f"{'='*60}\n")
                        else: 
                            print(f"\n{'='*60}")
                            print(f"[{timestamp_str}] ENGLISH DETECTED:")
                            print(f"  {text}")
                            print(f"  (Whisper detected: {detected_lang})")
                            
                            spanish_translation = translator.translate_to_spanish(text)
                            
                            if spanish_translation:
                                print(f"\n[{timestamp_str}] SPANISH TRANSLATION:")
                                print(f"  {spanish_translation}")
                                print(f"\n Speaking in Spanish...")
                                translator.speak_text(spanish_translation)
                                print("TTS completed")
                            print(f"{'='*60}\n")
                        
                        print("Ready for next input...\n")
                
                except KeyboardInterrupt:
                    print("\n\n Translator interrupted by user")
                except Exception as e:
                    print(f"\n Error in translator: {e}\n")
                finally:
                    translator.is_running = False
                    oled_control_owner = None
                    oled_control_lock.release()
                    print("OLED control: Released (detect resumed)\n")
                
                print("\n Returning to wake word detection...\n")
                print("="*60 + "\n")
                frame_count = 0 
    
    except KeyboardInterrupt:
        print("\n\n Goodbye!")
        print("JARVIS nursing assistant shutting down...")
        translator.is_running = False

if __name__ == "__main__":
    detect.oled_control_lock = oled_control_lock
    
    detect_thread = threading.Thread(target=detect.main, daemon=True)
    detect_thread.start()
    
    print("Detection system started in background thread")
    print("   (Initializing webcam and audio monitoring...)\n")
    
    max_wait = 10 
    waited = 0
    while detect.oled is None and waited < max_wait:
        time.sleep(0.5)
        waited += 0.5
    
    if detect.oled is None:
        print("Warning: OLED not initialized by detect module")
    else:
        import JARVIS as jarvis_module
        jarvis_module.oled = detect.oled
        print("OLED instance shared with JARVIS")
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n Shutting down all systems...")
    
    print("Main program exited. Detection thread will stop automatically.")

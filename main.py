#!/usr/bin/env python3
"""
JARVIS Nursing Assistant - Main Application
Wake words: "Jeff" for JARVIS, "Translate" for Spanish-English translator
"""

from JARVIS import create_jarvis
from translate import SpanishEnglishTranslator
import os
from datetime import datetime
import threading
import sys
import time
import detect


def listen_for_wake_words(jarvis, frame_duration=2):
    """
    Listen for wake words in speech.
    
    Args:
        jarvis: JARVIS instance
        frame_duration: Recording frame duration in seconds (default 2)
    
    Returns:
        "jeff" if JARVIS wake word detected
        "translate" if translator wake word detected
        None if no wake word detected
    """
    text = jarvis.listen_continuous(frame_duration=frame_duration, language="en-US")
    
    if text:
        text_lower = text.lower()
        if "jeff" in text_lower:
            return "jeff"
        elif "translate" in text_lower:
            return "translate"
    
    return None

def main():
    # Create JARVIS instance
    print("="*60)
    print("🩺 JARVIS - Nursing Assistant")
    print("="*60)
    print("\nInitializing JARVIS...")
    print("(Loading system instructions from jarvis_instructions.txt)")
    jarvis = create_jarvis()
    
    # Setup microphone (looks for PCM first, falls back to default)
    print("\n🔧 Setting up microphone...")
    jarvis.setup_microphone(device_name_prefix="PCM", auto_select=True)
    
    # Load context from output folder and knowledge base
    print("\n📚 Loading clinical context...")
    try:
        # Recursively add all .txt files from output folder
        if os.path.exists("output"):
            chunks = jarvis.add_context_from_directory("output", recursive=True)
            if chunks > 0:
                print(f"✅ Loaded clinical reports from output folder")
        
        # Add knowledge base if it exists
        if os.path.exists("jarvis_kb.txt"):
            jarvis.add_context_from_file("jarvis_kb.txt")
            print(f"✅ Loaded knowledge base")
    except Exception as e:
        print(f"⚠️ Warning loading context: {e}")
    
    # Initialize translator
    translator = SpanishEnglishTranslator()
    
    print("\n" + "="*60)
    print("🎙️  JARVIS Nursing Assistant - Voice Interface")
    print("="*60)
    print("\n✨ Wake Words:")
    print("   • Say 'JEFF' → Activate JARVIS nursing assistant")
    print("   • Say 'TRANSLATE' → Activate Spanish-English translator")
    print("\n💡 Features:")
    print("   - Auto-stop when you finish speaking")
    print("   - Voice responses with OpenAI TTS 🔊")
    print("   - JARVIS: Clinical context from reports 📚")
    print("   - Translator: Real-time Spanish ↔ English")
    print("   - Press Ctrl+C to exit")
    print("\n⚙️  Status: Ready - listening for wake words")
    print("="*60 + "\n")
    
    frame_count = 0
    try:
        while True:
            frame_count += 1
            print(f"🔵 Listening [Frame {frame_count}] - Say 'Jeff' or 'Translate'...")
            
            # Listen for wake words (2-second frame)
            wake_word = listen_for_wake_words(jarvis, frame_duration=2)
            
            if wake_word == "jeff":
                print("✅ Wake word 'Jeff' detected - JARVIS mode!")
                print("🎤 Listening for your question...\n")
                
                # Generate filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                audio_filename = f"jarvis_response_{timestamp}.mp3"
                
                # Now listen for the actual command with auto-stop
                response = jarvis.listen_and_ask(
                    auto_stop=True,           # Automatically stop when user stops talking
                    max_duration=30,          # Maximum 30 seconds
                    silence_duration=1.5,     # Stop after 1.5s of silence
                    use_rag=True,             # Use RAG with clinical context
                    speak_response=True,      # Speak the response out loud
                    save_audio_to=audio_filename  # Save audio to file
                )
                
                if response:
                    print(f"\n🤖 JARVIS: {response}")
                    print(f"🔊 (Response spoken and saved to {audio_filename})\n")
                else:
                    print("⚠️ Could not process your question. Please try again.\n")
                
                print("-" * 60 + "\n")
                frame_count = 0  # Reset frame counter after interaction
            
            elif wake_word == "translate":
                print("✅ Wake word 'Translate' detected - Translator mode!")
                print("🇪🇸↔️🇬🇧 Starting continuous translator...")
                print("   Person speaks English → Translated to Spanish + spoken")
                print("   Person speaks Spanish → Translated to English + spoken")
                print("\n   To exit, say any of:")
                print("   • 'stop translating' or 'end translating'")
                print("   • 'stop translation' or 'end translation'")
                print("   • Or press Ctrl+C\n")
                print("="*60 + "\n")
                
                # Enter continuous translator mode
                translator.is_running = True
                conversation_count = 0
                
                try:
                    while translator.is_running:
                        conversation_count += 1
                        print(f"🎤 [Turn {conversation_count}] Listening... Speak now!")
                        
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
                        ]
                        if any(phrase in text_lower for phrase in exit_phrases):
                            print("\n🛑 Exit command detected. Leaving translator mode...\n")
                            break
                        
                        timestamp_str = datetime.now().strftime("%H:%M:%S")
                        
                        # Detect language automatically and translate to the other
                        language = translator.detect_language(text, detected_lang)
                        
                        if language == "spanish":
                            # Spanish → English
                            print(f"\n{'='*60}")
                            print(f"[{timestamp_str}] 🇪🇸 SPANISH DETECTED:")
                            print(f"  {text}")
                            print(f"  (Whisper detected: {detected_lang})")
                            
                            english_translation = translator.translate_to_english(text)
                            
                            if english_translation:
                                print(f"\n[{timestamp_str}] 🇬🇧 ENGLISH TRANSLATION:")
                                print(f"  {english_translation}")
                                print(f"\n🔊 Speaking in English...")
                                translator.speak_text(english_translation)
                                print("✅ TTS completed")
                            print(f"{'='*60}\n")
                        else:  # English
                            # English → Spanish
                            print(f"\n{'='*60}")
                            print(f"[{timestamp_str}] 🇬🇧 ENGLISH DETECTED:")
                            print(f"  {text}")
                            print(f"  (Whisper detected: {detected_lang})")
                            
                            spanish_translation = translator.translate_to_spanish(text)
                            
                            if spanish_translation:
                                print(f"\n[{timestamp_str}] 🇪🇸 SPANISH TRANSLATION:")
                                print(f"  {spanish_translation}")
                                print(f"\n🔊 Speaking in Spanish...")
                                translator.speak_text(spanish_translation)
                                print("✅ TTS completed")
                            print(f"{'='*60}\n")
                        
                        print("🟢 Ready for next input...\n")
                
                except KeyboardInterrupt:
                    print("\n\n🛑 Translator interrupted by user")
                except Exception as e:
                    print(f"\n❌ Error in translator: {e}\n")
                
                translator.is_running = False
                print("\n🔙 Returning to wake word detection...\n")
                print("="*60 + "\n")
                frame_count = 0  # Reset frame counter after translator
    
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        print("JARVIS nursing assistant shutting down...")
        translator.is_running = False

if __name__ == "__main__":
    # Start detect in a separate thread
    detect_thread = threading.Thread(target=detect.main, daemon=True)
    detect_thread.start()
    
    print("🎥 Detection system started in background thread")
    print("   (Initializing webcam and audio monitoring...)\n")
    
    # Give detect thread time to initialize
    time.sleep(2)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down all systems...")
    
    print("✅ Main program exited. Detection thread will stop automatically.")

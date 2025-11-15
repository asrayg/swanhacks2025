#!/usr/bin/env python3
"""
Example script demonstrating speech-to-text functionality with JARVIS
Uses sounddevice for reliable microphone input with wake word detection
"""

from JARVIS import create_jarvis
import time
import os
from datetime import datetime

def print_microphone_info(jarvis):
    """Print information about available microphones"""
    print("\n" + "="*60)
    print("🎤 Microphone Information")
    print("="*60)
    
    try:
        # List all available input devices using sounddevice
        mic_list = jarvis.list_microphones()
        
        if mic_list:
            print(f"\n📋 Available microphones ({len(mic_list)}):")
            for mic in mic_list:
                print(f"  [{mic['index']}] {mic['name']}")
                print(f"      Sample Rate: {mic['sample_rate']} Hz")
                print(f"      Channels: {mic['channels']}")
            
            # Find PCM device
            pcm_device = None
            for mic in mic_list:
                if mic['name'].startswith('PCM'):
                    pcm_device = mic
                    break
            
            if pcm_device:
                print(f"\n✅ Found PCM microphone:")
                print(f"   Index: {pcm_device['index']}")
                print(f"   Name: {pcm_device['name']}")
            else:
                print(f"\n📌 No PCM device found, will use system default")
        else:
            print("\n⚠️ No microphones detected!")
    
    except Exception as e:
        print(f"\n❌ Error detecting microphones: {e}")
    
    print("="*60)

def listen_for_wake_word(jarvis, wake_word="jarvis", frame_duration=2):
    """
    Listen for the wake word in speech using overlapping frames.
    Records 2-second frames with 1-second stride (50% overlap).
    
    This means:
    - Frame 1: 0-2 seconds
    - Frame 2: 1-3 seconds (overlaps 1 second with Frame 1)
    - Frame 3: 2-4 seconds (overlaps 1 second with Frame 2)
    
    Args:
        jarvis: JARVIS instance
        wake_word: The wake word to listen for
        frame_duration: Recording frame duration in seconds (default 2)
    
    Returns:
        True if wake word detected, False otherwise
    """
    # Record the current frame (non-blocking for status updates)
    text = jarvis.listen_continuous(frame_duration=frame_duration, language="en-US")
    
    if text and wake_word.lower() in text.lower():
        return True
    return False

def main():
    # Create JARVIS instance
    print("Initializing JARVIS...")
    print("(Loading system instructions from jarvis_instructions.txt)")
    jarvis = create_jarvis()
    
    # Print microphone information
    print_microphone_info(jarvis)
    
    # Setup microphone (looks for PCM first, falls back to default)
    print("\n🔧 Setting up microphone...")
    jarvis.setup_microphone(device_name_prefix="PCM", auto_select=True)
    
    # Load context from output folder and knowledge base
    print("\n📚 Loading context...")
    try:
        # Recursively add all .txt files from output folder
        if os.path.exists("output"):
            chunks = jarvis.add_context_from_directory("output", recursive=True)
            if chunks > 0:
                print(f"✅ Loaded context from output folder")
        
        # Add knowledge base if it exists
        if os.path.exists("jarvis_kb.txt"):
            jarvis.add_context_from_file("jarvis_kb.txt")
            print(f"✅ Loaded jarvis_kb.txt")
    except Exception as e:
        print(f"⚠️ Warning loading context: {e}")
    
    print("\n" + "="*60)
    print("🎙️  JARVIS Wake Word Demo with Voice Response")
    print("="*60)
    print("\n✨ Say 'JARVIS' to activate the assistant")
    print("   Then speak your command")
    print("\n💡 Features:")
    print("   - Continuous listening with 2-second frames")
    print("   - Speak clearly and say 'JARVIS' to wake up")
    print("   - Auto-stop listening when you finish talking")
    print("   - OpenAI TTS voice responses 🔊")
    print("   - Each response saved as jarvis_response_TIMESTAMP.mp3 💾")
    print("   - Context loaded from output/ folder and jarvis_kb.txt 📚")
    print("   - Press Ctrl+C to exit")
    print("\n⚙️  Technical: Auto-stop after 1.5s of silence, OpenAI TTS-HD voice: 'alloy'")
    print("="*60 + "\n")
    
    frame_count = 0
    try:
        while True:
            frame_count += 1
            print(f"🔵 Listening [Frame {frame_count}] - 2 second window...")
            
            # Listen for wake word (2-second frame)
            if listen_for_wake_word(jarvis, wake_word="jarvis", frame_duration=2):
                print("✅ Wake word detected!")
                print("🎤 Listening for your command...\n")
                
                # Generate filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                audio_filename = f"jarvis_response_{timestamp}.mp3"
                
                # Now listen for the actual command with auto-stop
                response = jarvis.listen_and_ask(
                    auto_stop=True,           # Automatically stop when user stops talking
                    max_duration=30,          # Maximum 30 seconds
                    silence_duration=1.5,     # Stop after 1.5s of silence
                    use_rag=True,             # Use RAG if context is available
                    speak_response=True,      # Speak the response out loud
                    save_audio_to=audio_filename  # Save audio to file
                )
                
                if response:
                    print(f"\n🤖 JARVIS: {response}")
                    print(f"🔊 (Speaking response and saved to {audio_filename})\n")
                else:
                    print("⚠️ Could not process your command. Please try again.\n")
                
                print("-" * 60 + "\n")
                frame_count = 0  # Reset frame counter after interaction
    
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        print("JARVIS shutting down...")


if __name__ == "__main__":
    main()


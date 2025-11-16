from JARVIS import create_jarvis
import time
import os
from datetime import datetime

def print_microphone_info(jarvis):
    """Print information about available microphones"""
    print("\n" + "="*60)
    print("Microphone Information")
    print("="*60)
    
    try:
        mic_list = jarvis.list_microphones()
        
        if mic_list:
            print(f"\nAvailable microphones ({len(mic_list)}):")
            for mic in mic_list:
                print(f"  [{mic['index']}] {mic['name']}")
                print(f"      Sample Rate: {mic['sample_rate']} Hz")
                print(f"      Channels: {mic['channels']}")
            
            pcm_device = None
            for mic in mic_list:
                if mic['name'].startswith('PCM'):
                    pcm_device = mic
                    break
            
            if pcm_device:
                print(f"\n Found PCM microphone:")
                print(f"   Index: {pcm_device['index']}")
                print(f"   Name: {pcm_device['name']}")
            else:
                print(f"\n No PCM device found, will use system default")
        else:
            print("\n No microphones detected!")
    
    except Exception as e:
        print(f"\n Error detecting microphones: {e}")
    
    print("="*60)

def listen_for_wake_word(jarvis, wake_word="jarvis", frame_duration=2):
    text = jarvis.listen_continuous(frame_duration=frame_duration, language="en-US")
    
    if text and wake_word.lower() in text.lower():
        return True
    return False

def main():
    print("Initializing JARVIS...")
    print("(Loading system instructions from jarvis_instructions.txt)")
    jarvis = create_jarvis()
    
    print_microphone_info(jarvis)
    
    print("\n Setting up microphone...")
    jarvis.setup_microphone(device_name_prefix="PCM", auto_select=True)
    
    print("\n Loading context...")
    try:
        if os.path.exists("output"):
            chunks = jarvis.add_context_from_directory("output", recursive=True)
            if chunks > 0:
                print(f" Loaded context from output folder")
        
        if os.path.exists("jarvis_kb.txt"):
            jarvis.add_context_from_file("jarvis_kb.txt")
            print(f" Loaded jarvis_kb.txt")
    except Exception as e:
        print(f" Warning loading context: {e}")
    
    print("\n" + "="*60)
    print("JARVIS Wake Word Demo with Voice Response")
    print("="*60)
    print("\n Say 'JARVIS' to activate the assistant")
    print("   Then speak your command")
    print("\n Features:")
    print("   - Continuous listening with 2-second frames")
    print("   - Speak clearly and say 'JARVIS' to wake up")
    print("   - Auto-stop listening when you finish talking")
    print("   - OpenAI TTS voice responses")
    print("   - Each response saved as jarvis_response_TIMESTAMP.mp3 💾")
    print("   - Context loaded from output/ folder and jarvis_kb.txt 📚")
    print("   - Press Ctrl+C to exit")
    print("\n  Technical: Auto-stop after 1.5s of silence, OpenAI TTS-HD voice: 'alloy'")
    print("="*60 + "\n")
    
    frame_count = 0
    try:
        while True:
            frame_count += 1
            print(f"Listening [Frame {frame_count}] - 2 second window...")
            
            if listen_for_wake_word(jarvis, wake_word="jarvis", frame_duration=2):
                print("Wake word detected!")
                print("Listening for your command...\n")
                
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
                    print(f" (Speaking response and saved to {audio_filename})\n")
                else:
                    print("Could not process your command. Please try again.\n")
                
                print("-" * 60 + "\n")
                frame_count = 0 
    
    except KeyboardInterrupt:
        print("\n\n Goodbye!")
        print("JARVIS shutting down...")


if __name__ == "__main__":
    main()


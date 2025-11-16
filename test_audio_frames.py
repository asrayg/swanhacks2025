#!/usr/bin/env python3
"""
Test script to capture audio frames, save them, and transcribe them.
Similar to how main.py captures audio for wake word detection.
"""

import sounddevice as sd
import numpy as np
import wave
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
import os
import sys

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def setup_microphone():
    """Setup microphone - uses default device"""
    try:
        devices = sd.query_devices()
        default_input = sd.default.device[0]
        
        if default_input is not None:
            mic_device = devices[default_input]
            sample_rate = int(mic_device['default_samplerate'])
            print(f"\n✅ Using microphone:")
            print(f"   Device: {mic_device['name']}")
            print(f"   Index: {default_input}")
            print(f"   Sample rate: {sample_rate} Hz")
            print(f"   Channels: {mic_device['max_input_channels']}")
            return default_input, sample_rate
        else:
            print("\n✅ Using system default microphone")
            return None, 44100  # Default sample rate
    except Exception as e:
        print(f"\n❌ Error setting up microphone: {e}")
        return None, 44100


def record_frame(device_index, sample_rate, duration=2):
    """
    Record a single audio frame.
    
    Args:
        device_index: Microphone device index
        sample_rate: Sample rate in Hz
        duration: Duration in seconds
    
    Returns:
        numpy array of audio data
    """
    audio_data = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,  # Mono
        dtype='int16',
        device=device_index
    )
    sd.wait()  # Wait until recording is finished
    return audio_data


def save_audio(audio_data, filename, sample_rate):
    """Save audio data to WAV file"""
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 2 bytes for int16
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())


def transcribe_audio(audio_file):
    """Transcribe audio using OpenAI Whisper"""
    try:
        with open(audio_file, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json"
            )
        return transcript.text.strip(), transcript.language
    except Exception as e:
        print(f"   ⚠️  Transcription error: {e}")
        return None, None


def main():
    print("="*70)
    print("🎤 AUDIO FRAME CAPTURE & TRANSCRIPTION TEST")
    print("="*70)
    print("\nThis script will:")
    print("  1. Record 2-second audio frames (like main.py wake word detection)")
    print("  2. Save each frame to a WAV file")
    print("  3. Transcribe using OpenAI Whisper and print the text")
    print("\nPress Ctrl+C to stop\n")
    print("="*70 + "\n")
    
    # Create output directory for audio files
    output_dir = "test_audio_frames"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Audio files will be saved to: {output_dir}/\n")
    
    # Setup microphone
    print("🔧 Setting up microphone...")
    device_index, sample_rate = setup_microphone()
    
    print("\n" + "="*70)
    print("✅ Ready! Starting audio capture...")
    print("="*70 + "\n")
    
    frame_count = 0
    
    try:
        while True:
            frame_count += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
            frame_filename = f"{output_dir}/frame_{timestamp}_{frame_count:04d}.wav"
            
            print(f"🔵 Frame {frame_count} [{datetime.now().strftime('%H:%M:%S')}] - Recording 2 seconds...")
            
            # Record audio frame (2 seconds)
            audio_data = record_frame(device_index, sample_rate, duration=2)
            
            # Calculate audio energy (volume)
            energy = np.abs(audio_data).mean()
            
            # Save to file
            save_audio(audio_data, frame_filename, sample_rate)
            print(f"   💾 Saved: {frame_filename}")
            print(f"   📊 Audio level: {int(energy)} (threshold ~500 for speech)")
            
            # Transcribe if there's enough audio energy
            if energy > 100:  # Low threshold to catch quiet speech
                print(f"   ⚙️  Transcribing...")
                text, language = transcribe_audio(frame_filename)
                
                if text:
                    print(f"   🌍 Language: {language}")
                    print(f"   📝 Transcribed: \"{text}\"")
                else:
                    print(f"   📝 Transcribed: (empty/no speech)")
            else:
                print(f"   📝 (Skipping transcription - audio too quiet)")
            
            print()  # Blank line between frames
            
    except KeyboardInterrupt:
        print("\n\n🛑 Test stopped by user")
        print(f"\n📊 Summary:")
        print(f"   Total frames captured: {frame_count}")
        print(f"   Audio files saved to: {output_dir}/")
        print(f"   Total duration: {frame_count * 2} seconds")
        print("\n✅ Test complete!")


if __name__ == "__main__":
    main()


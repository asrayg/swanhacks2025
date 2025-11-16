#!/usr/bin/env python3
"""
Test script to verify plughw:2,0 microphone is working correctly.
This script lists all audio devices and records a short test from plughw:2,0.
"""

import sounddevice as sd
import numpy as np
import wave
import os

def list_devices():
    """List all available audio devices"""
    print("\n" + "="*70)
    print("📋 AVAILABLE AUDIO DEVICES")
    print("="*70)
    
    devices = sd.query_devices()
    
    print("\nINPUT DEVICES:")
    print("-"*70)
    for idx, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"  [{idx}] {device['name']}")
            print(f"      Max Input Channels: {device['max_input_channels']}")
            print(f"      Default Sample Rate: {device['default_samplerate']} Hz")
            print(f"      Host API: {device['hostapi']}")
            print()
    
    print("OUTPUT DEVICES:")
    print("-"*70)
    for idx, device in enumerate(devices):
        if device['max_output_channels'] > 0:
            print(f"  [{idx}] {device['name']}")
            print(f"      Max Output Channels: {device['max_output_channels']}")
            print()
    
    print("="*70 + "\n")

def find_device_plughw():
    """Find the device index for plughw:2,0"""
    devices = sd.query_devices()
    
    print("🔍 Searching for plughw:2,0...")
    
    # Look for device matching hw:2,0 patterns
    target_patterns = ['hw:2,0', 'plughw:2,0', 'hw:2', 'card 2']
    
    for idx, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            device_name = device['name'].lower()
            if any(pattern in device_name for pattern in target_patterns):
                print(f"✅ Found: [{idx}] {device['name']}")
                return idx, device
    
    # Try PulseAudio
    print("⚠️ Direct ALSA device not found, trying PulseAudio...")
    for idx, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            if 'pulse' in device['name'].lower():
                print(f"✅ Using PulseAudio: [{idx}] {device['name']}")
                return idx, device
    
    return None, None

def test_recording(device_idx=None, duration=3):
    """Test recording from the device"""
    try:
        devices = sd.query_devices()
        
        if device_idx is None:
            device_idx = sd.default.device[0]
        
        device = devices[device_idx]
        sample_rate = int(device['default_samplerate'])
        channels = min(2, device['max_input_channels'])
        
        print(f"\n🎤 Testing microphone recording...")
        print(f"   Device: {device['name']}")
        print(f"   Sample Rate: {sample_rate} Hz")
        print(f"   Channels: {channels}")
        print(f"   Duration: {duration} seconds")
        print(f"\n🔴 Recording... Speak now!")
        
        # Record audio
        audio_data = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=channels,
            dtype='int16',
            device=device_idx
        )
        sd.wait()
        
        print(f"✅ Recording completed!")
        
        # Analyze audio
        audio_max = np.abs(audio_data).max()
        audio_mean = np.abs(audio_data).mean()
        
        print(f"\n📊 Audio Analysis:")
        print(f"   Max amplitude: {audio_max}")
        print(f"   Mean amplitude: {audio_mean:.2f}")
        
        if audio_max < 100:
            print(f"   ⚠️ WARNING: Very low audio levels detected!")
            print(f"      This might indicate the wrong microphone or very quiet input.")
        elif audio_mean < 50:
            print(f"   ⚠️ WARNING: Low average audio levels.")
            print(f"      Try speaking louder or adjusting microphone gain.")
        else:
            print(f"   ✅ Audio levels look good!")
        
        # Save to file
        output_file = "test_recording.wav"
        
        # Convert stereo to mono if needed
        if channels == 2 and len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1).astype('int16')
            final_channels = 1
        else:
            final_channels = channels
        
        with wave.open(output_file, 'wb') as wf:
            wf.setnchannels(final_channels)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data.tobytes())
        
        print(f"\n💾 Recording saved to: {output_file}")
        print(f"   You can play it back with: aplay {output_file}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during recording: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "="*70)
    print("🎤 MICROPHONE TEST UTILITY - plughw:2,0")
    print("="*70)
    
    # List all devices
    list_devices()
    
    # Find plughw:2,0
    device_idx, device = find_device_plughw()
    
    if device_idx is None:
        print("❌ Could not find plughw:2,0")
        print("\n💡 Tips:")
        print("   1. Make sure the microphone is connected")
        print("   2. Check with: arecord -l")
        print("   3. Try: arecord -D plughw:2,0 -f cd test.wav")
        print("   4. Configure PulseAudio to use the correct device")
        return
    
    # Test recording
    print("\n" + "="*70)
    test_recording(device_idx, duration=3)
    print("="*70 + "\n")
    
    print("✅ Test complete!")
    print("\n💡 To use this device with the translator:")
    print("   - The translator is already configured to use plughw:2,0")
    print("   - If using PulseAudio, set it as default:")
    print("     pactl set-default-source 2")
    print()

if __name__ == "__main__":
    main()


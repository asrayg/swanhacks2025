#!/usr/bin/env python3
"""
SwanHacks 2025 - Face Recognition Demo
Simple demonstration of team member recognition for nurse AI assistant
"""

import cv2
import time
from lightweight_face_recognition import LightweightFaceRecognizer, capture_frame_pi


def print_banner():
    """Print a nice banner for the demo."""
    print("\n" + "="*70)
    print("  🏥 SwanHacks 2025 - AI Nurse Assistant Face Recognition Demo")
    print("="*70)
    print("\n  Team Members: Rahul, Justin, Arjun, Asray, Mitra")
    print("\n  This demo shows real-time face recognition running on Raspberry Pi")
    print("  without heavy libraries like dlib - perfect for edge deployment!")
    print("\n" + "="*70 + "\n")


def main():
    """Run the hackathon demo."""
    print_banner()
    
    # Initialize recognizer
    print("🔧 Initializing face recognition system...")
    print("   Method: Haar Cascade + LBPH (Lightweight & Fast)")
    
    recognizer = LightweightFaceRecognizer(
        detection_method="haar",  # Ultra-fast on Pi
        recognition_method="lbph",  # Good accuracy with speed
        scale_factor=0.5  # Process half-size for speed
    )
    
    # Load team member faces
    print("\n📸 Loading team member faces...")
    team_members = {
        "Rahul": "Rahul.png",
        "Justin": "Justin2.jpg", 
        "Arjun": "Arjun.png",
        "Asray": "Asray.png",
        "Mitra": "Mitra.png"
    }
    
    loaded = []
    for name, img_path in team_members.items():
        if recognizer.load_face_image(img_path, name):
            loaded.append(name)
    
    if len(loaded) == 0:
        print("❌ No faces loaded! Make sure image files exist.")
        return
    
    print(f"✅ Loaded {len(loaded)}/{len(team_members)} team members:")
    for name in loaded:
        print(f"   • {name}")
    
    # Train the recognizer
    print("\n🧠 Training face recognition model...")
    recognizer.train()
    
    # Optional: Save model for future use
    recognizer.save_model("swanhacks_team_faces.pkl")
    print("💾 Model saved for future use")
    
    # Start recognition loop
    print("\n" + "="*70)
    print("🎥 STARTING LIVE RECOGNITION")
    print("="*70)
    print("\nCamera will capture frames and identify team members in real-time.")
    print("Press Ctrl+C to stop.\n")
    
    frame_count = 0
    total_recognitions = {name: 0 for name in loaded}
    start_time = time.time()
    
    try:
        while True:
            # Capture from Pi camera
            frame = capture_frame_pi()
            
            if frame is None:
                print("⚠️  Camera capture failed, retrying...")
                time.sleep(0.5)
                continue
            
            frame_count += 1
            
            # Scale down for processing
            small_frame = cv2.resize(
                frame,
                None,
                fx=recognizer.scale_factor,
                fy=recognizer.scale_factor
            )
            
            # Detect faces
            faces = recognizer.detect_faces(small_frame)
            
            if len(faces) == 0:
                print(f"[Frame {frame_count:04d}] No faces detected")
            else:
                print(f"[Frame {frame_count:04d}] Detected {len(faces)} face(s):", end=" ")
                
                # Recognize each face
                for face_bbox in faces:
                    x, y, w, h = face_bbox
                    
                    # Scale back to original size
                    scale = 1.0 / recognizer.scale_factor
                    x, y, w, h = int(x*scale), int(y*scale), int(w*scale), int(h*scale)
                    
                    # Recognize
                    name, confidence = recognizer.recognize_face(frame, (x, y, w, h))
                    
                    # Update stats
                    if name != "Unknown" and name in total_recognitions:
                        total_recognitions[name] += 1
                    
                    # Print result
                    if name == "Unknown":
                        print(f"❓ Unknown ({confidence:.1%})", end=" ")
                    else:
                        emoji = "✅"
                        print(f"{emoji} {name} ({confidence:.1%})", end=" ")
                
                print()  # New line
            
            # Small delay
            time.sleep(0.4)
    
    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("🛑 DEMO STOPPED")
        print("="*70)
    
    # Print statistics
    elapsed_time = time.time() - start_time
    print(f"\n📊 Demo Statistics:")
    print(f"   Total frames processed: {frame_count}")
    print(f"   Duration: {elapsed_time:.1f} seconds")
    print(f"   Average FPS: {frame_count/elapsed_time:.2f}")
    print(f"\n👥 Recognition Count:")
    for name in sorted(total_recognitions.keys()):
        count = total_recognitions[name]
        if count > 0:
            print(f"   • {name}: {count} times")
    
    print("\n✅ Demo complete! Thank you for watching!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()


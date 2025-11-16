#!/usr/bin/env python3
"""
SwanHacks 2025 - Face Recognition Demo (WEBCAM VERSION)
Test on your laptop before deploying to Raspberry Pi
"""

import cv2
import time
from lightweight_face_recognition import LightweightFaceRecognizer


def print_banner():
    """Print a nice banner for the demo."""
    print("\n" + "="*70)
    print("  🏥 SwanHacks 2025 - AI Nurse Assistant Face Recognition Demo")
    print("  💻 WEBCAM VERSION - Testing on Laptop")
    print("="*70)
    print("\n  Team Members: Rahul, Justin, Arjun, Asray, Mitra")
    print("\n  This demo uses your regular webcam for testing")
    print("  Press 'q' to quit")
    print("\n" + "="*70 + "\n")


def main():
    """Run the webcam demo."""
    print_banner()
    
    # Initialize recognizer
    print("🔧 Initializing face recognition system...")
    print("   Method: Haar Cascade + LBPH (Lightweight & Fast)")
    
    recognizer = LightweightFaceRecognizer(
        detection_method="haar",  # Ultra-fast
        recognition_method="lbph",  # Good accuracy
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
        print("   Looking for: Rahul.png, Justin.png, Arjun.png, Asray.png, Mitra.png")
        return
    
    print(f"✅ Loaded {len(loaded)}/{len(team_members)} team members:")
    for name in loaded:
        print(f"   • {name}")
    
    # Train the recognizer
    print("\n🧠 Training face recognition model...")
    recognizer.train()
    print("✅ Training complete!")
    
    # Optional: Save model for future use
    recognizer.save_model("swanhacks_team_faces.pkl")
    print("💾 Model saved for future use")
    
    # Open webcam
    print("\n🎥 Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Could not open webcam!")
        print("   Make sure your webcam is connected and not in use by another app.")
        return
    
    # Set camera resolution (optional, adjust if needed)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("✅ Webcam opened successfully!")
    
    # Start recognition loop
    print("\n" + "="*70)
    print("🎥 STARTING LIVE RECOGNITION")
    print("="*70)
    print("\nShowing live video with face recognition.")
    print("Press 'q' to quit.\n")
    
    frame_count = 0
    total_recognitions = {name: 0 for name in loaded}
    start_time = time.time()
    
    try:
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            
            if not ret:
                print("⚠️  Failed to grab frame from webcam")
                break
            
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
            
            # Process each face
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
                
                # Draw rectangle around face
                if name == "Unknown":
                    color = (0, 0, 255)  # Red for unknown
                else:
                    color = (0, 255, 0)  # Green for recognized
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Draw name and confidence
                label = f"{name} ({confidence:.1%})"
                
                # Background for text (makes it readable)
                (text_width, text_height), _ = cv2.getTextSize(
                    label, 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    2
                )
                cv2.rectangle(
                    frame,
                    (x, y - text_height - 10),
                    (x + text_width, y),
                    color,
                    -1  # Filled rectangle
                )
                
                # Text
                cv2.putText(
                    frame,
                    label,
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),  # White text
                    2
                )
            
            # Add info overlay
            info_text = f"Frame: {frame_count} | Faces: {len(faces)} | Press 'q' to quit"
            cv2.putText(
                frame,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            
            # Display the frame
            cv2.imshow('Face Recognition - SwanHacks 2025', frame)
            
            # Check for quit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n🛑 Quit key pressed...")
                break
    
    except KeyboardInterrupt:
        print("\n\n🛑 Interrupted by user (Ctrl+C)...")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Print statistics
    elapsed_time = time.time() - start_time
    print("\n" + "="*70)
    print("📊 Demo Statistics")
    print("="*70)
    print(f"   Total frames processed: {frame_count}")
    print(f"   Duration: {elapsed_time:.1f} seconds")
    print(f"   Average FPS: {frame_count/elapsed_time:.2f}")
    print(f"\n👥 Recognition Count:")
    
    recognized_anyone = False
    for name in sorted(total_recognitions.keys()):
        count = total_recognitions[name]
        if count > 0:
            print(f"   • {name}: {count} times")
            recognized_anyone = True
    
    if not recognized_anyone:
        print("   (No team members were recognized)")
    
    print("\n✅ Demo complete! Ready to deploy to Raspberry Pi!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()


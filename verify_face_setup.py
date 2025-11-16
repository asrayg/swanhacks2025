#!/usr/bin/env python3
"""
Quick verification script to check if face recognition setup is ready.
Run this before your demo to catch any issues early!
"""

import sys
import os


def check_module(module_name, package_name=None):
    """Check if a Python module can be imported."""
    if package_name is None:
        package_name = module_name
    
    try:
        __import__(module_name)
        print(f"✅ {package_name} is installed")
        return True
    except ImportError:
        print(f"❌ {package_name} is NOT installed")
        print(f"   Install with: pip3 install {package_name}")
        return False


def check_opencv_contrib():
    """Check if OpenCV contrib modules (including face) are available."""
    try:
        import cv2
        # Try to access the face module
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        print("✅ OpenCV face module (contrib) is available")
        return True
    except AttributeError:
        print("❌ OpenCV face module NOT available")
        print("   Install with: pip3 install opencv-contrib-python")
        return False
    except Exception as e:
        print(f"⚠️  Error checking OpenCV face module: {e}")
        return False


def check_files():
    """Check if face image files exist."""
    required_files = [
        "Rahul.png",
        "Justin.png",
        "Arjun.png",
        "Asray.png",
        "Mitra.png"
    ]
    
    all_exist = True
    print("\n📸 Checking face image files:")
    
    for filename in required_files:
        if os.path.exists(filename):
            print(f"   ✅ {filename}")
        else:
            print(f"   ❌ {filename} - NOT FOUND")
            all_exist = False
    
    return all_exist


def check_camera():
    """Check if camera is accessible."""
    print("\n📷 Checking camera access:")
    
    # Check if we're on a Raspberry Pi
    try:
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read()
            if 'Raspberry Pi' in model:
                print(f"   🥧 Detected: {model.strip()}")
                
                # Check if rpicam-still is available
                import subprocess
                result = subprocess.run(
                    ['which', 'rpicam-still'],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    print("   ✅ rpicam-still command found")
                    return True
                else:
                    print("   ❌ rpicam-still not found")
                    print("      Install with: sudo apt install libcamera-apps")
                    return False
    except FileNotFoundError:
        print("   ℹ️  Not running on Raspberry Pi (or can't detect)")
        print("   Will try to use regular webcam (cv2.VideoCapture)")
        
        # Try regular webcam
        try:
            import cv2
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                print("   ✅ Webcam detected (cv2.VideoCapture)")
                cap.release()
                return True
            else:
                print("   ⚠️  Could not open webcam")
                return False
        except Exception as e:
            print(f"   ⚠️  Error checking webcam: {e}")
            return False
    
    return True


def main():
    """Run all verification checks."""
    print("="*60)
    print("🔍 Face Recognition Setup Verification")
    print("="*60)
    
    print("\n📦 Checking Python packages:")
    
    checks = []
    
    # Check core dependencies
    checks.append(check_module("cv2", "opencv-python"))
    checks.append(check_module("numpy", "numpy"))
    checks.append(check_opencv_contrib())
    
    # Check face image files
    checks.append(check_files())
    
    # Check camera
    checks.append(check_camera())
    
    # Check if our scripts exist
    print("\n📝 Checking scripts:")
    scripts = [
        "lightweight_face_recognition.py",
        "demo_face_recognition.py"
    ]
    
    for script in scripts:
        if os.path.exists(script):
            print(f"   ✅ {script}")
        else:
            print(f"   ❌ {script} - NOT FOUND")
            checks.append(False)
    
    # Final summary
    print("\n" + "="*60)
    if all(checks):
        print("🎉 ALL CHECKS PASSED!")
        print("You're ready to run the face recognition demo.")
        print("\nRun: python3 demo_face_recognition.py")
    else:
        print("⚠️  SOME CHECKS FAILED")
        print("Please fix the issues above before running the demo.")
        sys.exit(1)
    print("="*60)


if __name__ == "__main__":
    main()


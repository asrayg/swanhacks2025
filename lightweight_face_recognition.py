#!/usr/bin/env python3
"""
Lightweight Face Recognition for Raspberry Pi
Uses OpenCV without dlib - perfect for resource-constrained devices

Three modes available:
1. HAAR_CASCADE - Ultra lightweight (fastest)
2. DNN_CAFFE - More accurate, still Pi-friendly
3. LBPH - Local Binary Patterns Histogram (good middle ground)
"""

import cv2
import numpy as np
import os
import pickle
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class LightweightFaceRecognizer:
    """
    Lightweight face recognition system optimized for Raspberry Pi.
    No dlib or heavy dependencies required!
    """
    
    # Detection methods
    HAAR_CASCADE = "haar"
    DNN_CAFFE = "dnn"
    
    # Recognition methods
    LBPH = "lbph"  # Local Binary Patterns Histogram
    TEMPLATE = "template"  # Simple template matching
    
    def __init__(
        self,
        detection_method: str = "haar",
        recognition_method: str = "lbph",
        scale_factor: float = 0.5  # Scale down frames for faster processing
    ):
        """
        Initialize the face recognizer.
        
        Args:
            detection_method: "haar" (fastest) or "dnn" (more accurate)
            recognition_method: "lbph" or "template"
            scale_factor: Scale factor for frame processing (0.5 = half size)
        """
        self.detection_method = detection_method
        self.recognition_method = recognition_method
        self.scale_factor = scale_factor
        
        # Initialize face detector
        if detection_method == self.HAAR_CASCADE:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                raise RuntimeError(f"Failed to load Haar Cascade from {cascade_path}")
            print("✅ Loaded Haar Cascade face detector")
            
        elif detection_method == self.DNN_CAFFE:
            # Download models if not present
            self._ensure_dnn_models()
            model_file = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
            config_file = "models/deploy.prototxt"
            self.dnn_net = cv2.dnn.readNetFromCaffe(config_file, model_file)
            print("✅ Loaded DNN face detector (Caffe)")
        
        # Initialize face recognizer
        if recognition_method == self.LBPH:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create(
                radius=1,
                neighbors=8,
                grid_x=8,
                grid_y=8
            )
            print("✅ Initialized LBPH face recognizer")
        
        self.known_faces = {}  # {name: face_template/encoding}
        self.label_to_name = {}
        self.trained = False
    
    def _ensure_dnn_models(self):
        """Download DNN models if not present."""
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        model_file = models_dir / "res10_300x300_ssd_iter_140000_fp16.caffemodel"
        config_file = models_dir / "deploy.prototxt"
        
        if not model_file.exists():
            print("📥 Downloading DNN face detection model...")
            url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
            import urllib.request
            urllib.request.urlretrieve(url, str(model_file))
            print("✅ Downloaded model file")
        
        if not config_file.exists():
            print("📥 Downloading DNN config...")
            url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
            import urllib.request
            urllib.request.urlretrieve(url, str(config_file))
            print("✅ Downloaded config file")
    
    def detect_faces_haar(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using Haar Cascades (fastest method)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return [(x, y, w, h) for (x, y, w, h) in faces]
    
    def detect_faces_dnn(self, frame: np.ndarray, confidence_threshold: float = 0.5) -> List[Tuple[int, int, int, int]]:
        """Detect faces using DNN (more accurate)."""
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )
        
        self.dnn_net.setInput(blob)
        detections = self.dnn_net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                faces.append((x1, y1, x2 - x1, y2 - y1))
        
        return faces
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using the configured method."""
        if self.detection_method == self.HAAR_CASCADE:
            return self.detect_faces_haar(frame)
        elif self.detection_method == self.DNN_CAFFE:
            return self.detect_faces_dnn(frame)
        else:
            raise ValueError(f"Unknown detection method: {self.detection_method}")
    
    def load_face_image(self, image_path: str, name: str) -> bool:
        """
        Load a face image for a person.
        
        Args:
            image_path: Path to the image file
            name: Name of the person
        
        Returns:
            True if successful
        """
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return False
        
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Failed to load image: {image_path}")
            return False
        
        # Detect face in image
        faces = self.detect_faces(img)
        
        if len(faces) == 0:
            print(f"⚠️  No face detected in {image_path}")
            return False
        
        if len(faces) > 1:
            print(f"⚠️  Multiple faces detected in {image_path}, using first one")
        
        # Get the first face
        x, y, w, h = faces[0]
        face_roi = img[y:y+h, x:x+w]
        
        # Store face data
        if self.recognition_method == self.TEMPLATE:
            # Store normalized face template
            face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            face_resized = cv2.resize(face_gray, (100, 100))
            self.known_faces[name] = face_resized
        else:
            # For LBPH, we'll store the face for training
            if name not in self.known_faces:
                self.known_faces[name] = []
            face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            self.known_faces[name].append(face_gray)
        
        print(f"✅ Loaded face for {name} from {image_path}")
        return True
    
    def train(self):
        """Train the face recognizer (for LBPH method)."""
        if self.recognition_method != self.LBPH:
            print("⚠️  Training only needed for LBPH method")
            return
        
        if not self.known_faces:
            print("❌ No faces loaded. Use load_face_image() first.")
            return
        
        # Prepare training data
        faces = []
        labels = []
        
        for label_id, (name, face_list) in enumerate(self.known_faces.items()):
            self.label_to_name[label_id] = name
            for face in face_list:
                faces.append(face)
                labels.append(label_id)
        
        # Train the recognizer
        print(f"🔄 Training LBPH recognizer with {len(faces)} face samples...")
        self.recognizer.train(faces, np.array(labels))
        self.trained = True
        print("✅ Training complete!")
    
    def recognize_face(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Tuple[str, float]:
        """
        Recognize a face in the given bounding box.
        
        Args:
            frame: The full frame
            face_bbox: (x, y, w, h) bounding box of the face
        
        Returns:
            (name, confidence) tuple. name is "Unknown" if not recognized.
        """
        x, y, w, h = face_bbox
        face_roi = frame[y:y+h, x:x+w]
        face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        if self.recognition_method == self.TEMPLATE:
            return self._recognize_template(face_gray)
        elif self.recognition_method == self.LBPH:
            return self._recognize_lbph(face_gray)
        else:
            return ("Unknown", 0.0)
    
    def _recognize_template(self, face_gray: np.ndarray) -> Tuple[str, float]:
        """Recognize using template matching (simple but works)."""
        if not self.known_faces:
            return ("Unknown", 0.0)
        
        face_resized = cv2.resize(face_gray, (100, 100))
        
        best_match = "Unknown"
        best_score = 0.0
        threshold = 0.5  # Similarity threshold
        
        for name, template in self.known_faces.items():
            # Use normalized correlation
            result = cv2.matchTemplate(face_resized, template, cv2.TM_CCOEFF_NORMED)
            score = result[0][0]
            
            if score > best_score and score > threshold:
                best_score = score
                best_match = name
        
        return (best_match, best_score)
    
    def _recognize_lbph(self, face_gray: np.ndarray) -> Tuple[str, float]:
        """Recognize using LBPH recognizer."""
        if not self.trained:
            return ("Unknown", 0.0)
        
        label, confidence = self.recognizer.predict(face_gray)
        
        # LBPH returns lower confidence for better matches
        # Threshold: < 50 is good, < 80 is acceptable
        if confidence < 80:
            name = self.label_to_name.get(label, "Unknown")
            # Convert to 0-1 scale (inverse)
            normalized_confidence = max(0, 1 - (confidence / 100))
            return (name, normalized_confidence)
        else:
            return ("Unknown", 0.0)
    
    def save_model(self, path: str = "face_recognizer.pkl"):
        """Save the trained model."""
        if self.recognition_method == self.LBPH:
            # Save LBPH model
            model_path = path.replace('.pkl', '.yml')
            self.recognizer.save(model_path)
            # Save label mapping
            with open(path, 'wb') as f:
                pickle.dump(self.label_to_name, f)
            print(f"✅ Model saved to {model_path} and {path}")
        else:
            # Save templates
            with open(path, 'wb') as f:
                pickle.dump(self.known_faces, f)
            print(f"✅ Face templates saved to {path}")
    
    def load_model(self, path: str = "face_recognizer.pkl"):
        """Load a trained model."""
        if self.recognition_method == self.LBPH:
            model_path = path.replace('.pkl', '.yml')
            if os.path.exists(model_path):
                self.recognizer.read(model_path)
                self.trained = True
            with open(path, 'rb') as f:
                self.label_to_name = pickle.load(f)
            print(f"✅ Model loaded from {model_path} and {path}")
        else:
            with open(path, 'rb') as f:
                self.known_faces = pickle.load(f)
            print(f"✅ Face templates loaded from {path}")


def capture_frame_pi(path="/dev/shm/frame.jpg"):
    """Capture a frame using Raspberry Pi camera."""
    cmd = [
        "rpicam-still",
        "-t", "1",
        "--width", "640",
        "--height", "480",
        "-n",
        "-o", path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return cv2.imread(path)


def main_demo():
    """Demo script for Raspberry Pi."""
    print("="*60)
    print("🎭 Lightweight Face Recognition for Raspberry Pi")
    print("="*60)
    
    # Choose method based on your needs:
    # - "haar" + "template": FASTEST, good for demos
    # - "haar" + "lbph": Fast and more accurate
    # - "dnn" + "lbph": Best accuracy, still Pi-friendly
    
    recognizer = LightweightFaceRecognizer(
        detection_method="haar",  # Change to "dnn" for better accuracy
        recognition_method="lbph",  # or "template" for simplicity
        scale_factor=0.5  # Process half-size frames for speed
    )
    
    # Load face images
    print("\n📸 Loading face images...")
    face_images = {
        "Rahul": "Rahul.png",
        "Justin": "Justin.png",
        "Arjun": "Arjun.png",
        "Asray": "Asray.png",
        "Mitra": "Mitra.png"
    }
    
    loaded_count = 0
    for name, img_path in face_images.items():
        if recognizer.load_face_image(img_path, name):
            loaded_count += 1
    
    print(f"\n✅ Loaded {loaded_count}/{len(face_images)} faces")
    
    # Train if using LBPH
    if recognizer.recognition_method == recognizer.LBPH:
        recognizer.train()
    
    # Optional: Save model for faster startup next time
    # recognizer.save_model("team_faces.pkl")
    
    print("\n🎥 Starting face recognition...")
    print("Press Ctrl+C to stop\n")
    
    import time
    frame_count = 0
    
    try:
        while True:
            # Capture frame from Pi camera
            frame = capture_frame_pi()
            
            if frame is None:
                print("⚠️  Failed to capture frame")
                time.sleep(0.5)
                continue
            
            # Scale down for faster processing
            if recognizer.scale_factor != 1.0:
                small_frame = cv2.resize(
                    frame,
                    None,
                    fx=recognizer.scale_factor,
                    fy=recognizer.scale_factor
                )
            else:
                small_frame = frame
            
            # Detect faces
            faces = recognizer.detect_faces(small_frame)
            
            # Recognize each face
            recognized_names = []
            for face_bbox in faces:
                x, y, w, h = face_bbox
                
                # Scale back to original coordinates
                if recognizer.scale_factor != 1.0:
                    scale = 1.0 / recognizer.scale_factor
                    x, y, w, h = int(x*scale), int(y*scale), int(w*scale), int(h*scale)
                
                # Recognize face
                name, confidence = recognizer.recognize_face(frame, (x, y, w, h))
                recognized_names.append((name, confidence))
                
                # Draw on frame (optional if you have display)
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                label = f"{name} ({confidence:.2f})"
                cv2.putText(
                    frame,
                    label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )
            
            # Print results
            if recognized_names:
                frame_count += 1
                print(f"[Frame {frame_count}] Detected:", end=" ")
                for name, conf in recognized_names:
                    print(f"{name} ({conf:.2%})", end=" ")
                print()
            
            # Optional: Display frame if you have a display attached
            # cv2.imshow("Face Recognition", frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            
            # Small delay to avoid hammering the camera
            time.sleep(0.3)
    
    except KeyboardInterrupt:
        print("\n\n👋 Stopping face recognition...")
    
    # Cleanup
    # cv2.destroyAllWindows()
    print("✅ Done!")


if __name__ == "__main__":
    main_demo()


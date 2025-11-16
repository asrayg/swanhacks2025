import face_recognition
import cv2
import numpy as np
import subprocess
import time

def capture_frame(path="/dev/shm/frame.jpg"):
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


rahul_image = face_recognition.load_image_file("Rahul.png")
rahul_face_encoding = face_recognition.face_encodings(rahul_image)[0]

justin_image = face_recognition.load_image_file("Justin.png")
justin_face_encoding = face_recognition.face_encodings(justin_image)[0]

arjun_image = face_recognition.load_image_file("Arjun.png")
arjun_face_encoding = face_recognition.face_encodings(arjun_image)[0]

asray_image = face_recognition.load_image_file("Asray.png")
asray_face_encoding = face_recognition.face_encodings(asray_image)[0]

mitra_image = face_recognition.load_image_file("Mitra.png")
mitra_face_encoding = face_recognition.face_encodings(mitra_image)[0]

known_face_encodings = [
    rahul_face_encoding,
    justin_face_encoding,
    arjun_face_encoding,
    asray_face_encoding,
    mitra_face_encoding
]

known_face_names = [
    "Rahul",
    "Justin",
    "Arjun",
    "Asray",
    "Mitra"
]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

print("📸 Starting Raspberry Pi face recognition")
print("Press Ctrl+C to stop")

while True:
    # Capture from Pi camera
    frame = capture_frame()

    if frame is None:
        print("⚠️ Could not capture frame — retrying...")
        time.sleep(0.2)
        continue

    if process_this_frame:
        # Speed up by shrinking frame
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert to RGB for face_recognition
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect locations + encodings
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Print results to terminal
    for name in face_names:
        print(f"🎯 Detected: {name}")

    # OPTIONAL: Draw on image if GUI available
    # (comment out if headless)
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4; right *= 4; bottom *= 4; left *= 4
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, bottom-6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255), 1)

    # If running headless comment this out:
    # cv2.imshow("Pi Face Recognition", frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break

    # small delay to avoid hammering the camera
    time.sleep(0.2)

# Cleanup
# cv2.destroyAllWindows()

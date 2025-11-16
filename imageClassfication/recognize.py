import face_recognition
import cv2
import numpy as np

# --- 1. Load Known Faces and Encodings ---

# Load Rahul's image and learn how to recognize it.
rahul_image = face_recognition.load_image_file("Rahul.png")
rahul_face_encoding = face_recognition.face_encodings(rahul_image)[0]

# Load Justin's image and learn how to recognize it.
justin_image = face_recognition.load_image_file("Justin.png")
justin_face_encoding = face_recognition.face_encodings(justin_image)[0]

# Load Arjun's image and learn how to recognize it.
arjun_image = face_recognition.load_image_file("Arjun.png")
arjun_face_encoding = face_recognition.face_encodings(arjun_image)[0]

# Load Asray's image and learn how to recognize it.
asray_image = face_recognition.load_image_file("Asray.png")
asray_face_encoding = face_recognition.face_encodings(asray_image)[0]

#Load Mitra's image and learn how to recognize it.
mitra_image = face_recognition.load_image_file("Mitra.png")
mitra_face_encoding = face_recognition.load_image_file("Mitra.png")

# Create arrays of your known face encodings and their names
known_face_encodings = [
    rahul_face_encoding,
    justin_face_encoding,
    arjun_face_encoding,
    asray_face_encoding
]
known_face_names = [
    "Rahul",
    "Justin",
    "Arjun",
    "Asray"
]

# --- 2. Initialize Variables for Live Video ---

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

print("Starting video feed. Press 'q' to quit.")

# --- 3. Start the Live Video Loop ---

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    
    # For performance, only process every other frame
    if process_this_frame:
        # Resize frame to 1/4 size for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        
        # Convert the image from BGR color (which OpenCV uses) to RGB color
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Find all the faces and face encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown" # Default name if no match

            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # --- 4. Display the Results ---

    # Loop through each face found in the frame
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations (since we processed a 1/4 size image)
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        # (BGR color format for OpenCV)
        if name == "Unknown":
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2) # Red
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        else:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2) # Green
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            print(name)
            
        # Draw a label with a name below the face
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
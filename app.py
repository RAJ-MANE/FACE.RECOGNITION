from flask import Flask, render_template, Response, stream_with_context
import cv2
import face_recognition
import time

app = Flask(__name__)

# Load the single reference image
img1 = face_recognition.load_image_file("raj.jpg")
rgb_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img_encodings1 = face_recognition.face_encodings(rgb_img1)[0]

# Only one known face is used now
known_face_encodings = [img_encodings1]
known_face_names = ["Raj"]

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

verified_user = ""

def generate_frames():
    global verified_user
    frame_count = 0
    scale_factor = 0.5  
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1

        if frame_count % 2 == 0:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small_frame = cv2.resize(rgb_frame, (0, 0), fx=scale_factor, fy=scale_factor)

        face_locations_small = face_recognition.face_locations(small_frame, model="hog")
        face_encodings_small = face_recognition.face_encodings(small_frame, face_locations_small)

        face_locations = [
            (int(top/scale_factor), int(right/scale_factor), int(bottom/scale_factor), int(left/scale_factor))
            for (top, right, bottom, left) in face_locations_small
        ]

        verified_user = ""
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings_small):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            if name != "Unknown":
                verified_user = f"Verified User: {name}"
            
            # Draw a thin rectangle around the detected face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 1)
            
            # Calculate text size and position for centered text
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.5
            thickness = 1
            text_size, _ = cv2.getTextSize(name, font, font_scale, thickness)
            text_width = text_size[0]
            text_x = left + (right - left - text_width) // 2
            text_y = bottom - 6
            
            cv2.rectangle(frame, (text_x - 2, text_y - text_size[1] - 2),
                          (text_x + text_width + 2, text_y + 2), (0, 0, 255), cv2.FILLED)
            
            cv2.putText(frame, name, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/user_status')
def user_status():
    def event_stream():
        while True:
            time.sleep(1)
            yield f"data: {verified_user}\n\n"
    return Response(stream_with_context(event_stream()), mimetype="text/event-stream")

if __name__ == '__main__':
    app.run(debug=True)

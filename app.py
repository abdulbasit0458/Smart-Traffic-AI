from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO
import pytesseract
import re
import csv
from datetime import datetime

app = Flask(__name__)

# 🔥 Tesseract path (check correct)
pytesseract.pytesseract.tesseract_cmd = r"D:\New folder\tesseract.exe"

# 🔥 model
model = YOLO("runs/detect/license_plate_model-5/weights/best.pt")

# 🔥 video
video_path = r"D:\AI HACKATHON\smart-traffic-anpr\data\videos\traffic.mp4"
cap = cv2.VideoCapture(video_path)

last_plate = ""

FILE_NAME = "plates_log.csv"

# ✅ save function
def save_plate(text):
    text = text.strip()

    if len(text) < 5:
        return

    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(FILE_NAME, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([text, time_now])


# ✅ video generator
def generate_frames():
    global last_plate

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, conf=0.10)

        for r in results:
            for box in r.boxes:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                plate = frame[y1:y2, x1:x2]

                if plate.size == 0:
                    continue

                gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, None, fx=2, fy=2)

                text = pytesseract.image_to_string(gray, config="--psm 7")

                text = re.sub(r'[^A-Z0-9]', '', text.upper())

                if len(text) >= 5 and text != last_plate:
                    save_plate(text)
                    last_plate = text

                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, text, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0,255,0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# ✅ 🔥 UPDATED INDEX (TABLE SHOW)
def index():
    plates = []

    try:
        with open("plates_log.csv", "r") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                plates.append(row)
    except:
        pass

    return render_template("index.html", plates=plates)


# routes
app.add_url_rule('/', 'index', index)

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
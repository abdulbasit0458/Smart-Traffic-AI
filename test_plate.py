import cv2
from ultralytics import YOLO
import pytesseract
import re

from logger import save_plate

# 🔥 Tesseract path (CHECK THIS)
pytesseract.pytesseract.tesseract_cmd = r"D:\New folder\tesseract.exe"

# 🔥 MODEL
model = YOLO("runs/detect/license_plate_model-5/weights/best.pt")

# 🔥 VIDEO PATH
video_path = r"D:\AI HACKATHON\smart-traffic-anpr\data\videos\traffic.mp4"

cap = cv2.VideoCapture(video_path)

last_plate = ""   # duplicate control

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.10)

    for r in results:
        for box in r.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # crop plate
            plate = frame[y1:y2, x1:x2]

            if plate.size == 0:
                continue

            # preprocessing (IMPORTANT)
            gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=2, fy=2)
            gray = cv2.GaussianBlur(gray, (3,3), 0)

            # OCR
            text = pytesseract.image_to_string(gray, config="--psm 7")

            # 🔥 CLEAN TEXT (REMOVE GARBAGE)
            text = re.sub(r'[^A-Z0-9]', '', text.upper())

            # filter noise
            if len(text) < 5:
                continue

            # 🔥 REMOVE DUPLICATES
            if text != last_plate:
                save_plate(text)
                last_plate = text

            # draw box
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

            cv2.putText(frame, text, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0,255,0), 2)

    cv2.imshow("ANPR SYSTEM", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
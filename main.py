import cv2
import numpy as np
from util import get_limits

green = [0, 255, 0]  # The BGR colour space for Green
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lowerLimit, upperLimit = get_limits(green)
    mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)

    # Find bounding box of all non-zero pixels in the mask using OpenCV
    coords = cv2.findNonZero(mask)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        x1, y1, x2, y2 = x, y, x + w, y + h

        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

    cv2.imshow('Color Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
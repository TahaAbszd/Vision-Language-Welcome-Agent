import cv2 # type: ignore
import numpy as np # type: ignore
from ultralytics import YOLO # type: ignore
import math

# Load YOLO Pose model
model = YOLO("yolo26n-pose.pt")  

cap = cv2.VideoCapture(0)

def dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def get_angle(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1, 1)))

def detect_gesture(kp):

    ls = kp[5]
    rs = kp[6]
    le = kp[7]
    re = kp[8]
    lw = kp[9]
    rw = kp[10]

    shoulder_width = dist(ls, rs)
    center = ((ls[0]+rs[0])/2, (ls[1]+rs[1])/2)

    if rw[1] < rs[1]:
        return "raise right hand"

    if lw[1] < ls[1]:
        return "raise left hand"

    if rw[1] < rs[1] and lw[1] < ls[1]:
        return "both hands up"

    if dist(lw, rw) > 2.2 * shoulder_width:
        return "arms open"

    if dist(lw, rw) < 40:
        return "clap"

    angle = get_angle(rs, re, rw)
    if angle > 160 and rw[0] > rs[0]:
        return "pointing"

    if dist(lw, rw) < shoulder_width * 1.2 and lw[1] > ls[1]:
        return "hug"

    return "neutral"


while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for r in results:
        if r.keypoints is None:
            continue

        for person_kp in r.keypoints.xy.cpu().numpy():

            gesture = detect_gesture(person_kp)

            cv2.putText(frame, gesture, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
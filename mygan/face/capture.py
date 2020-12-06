import cv2
from mygan.face.face_recognizer import FaceRecognizer
from mygan import TmpFilePath
import os
import time

FACE_MARGIN_SCALE = 1.6
SAVE_IMAGE_SIZE = 128


def capture():
    video = cv2.VideoCapture(0)
    recognizer = FaceRecognizer()

    save_path = os.path.join(TmpFilePath, "face")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save = False

    while True:
        ret, frame = video.read()
        recognized_results = recognizer.recognize(frame, min_size=(128, 128))

        if len(recognized_results) > 0:
            fh, fw, _ = frame.shape
            x, y, w, h = recognized_results[0]
            sw = int(w * FACE_MARGIN_SCALE)
            sh = int(h * FACE_MARGIN_SCALE)
            sx = x - (sw - w) // 2
            sy = y - int((sh - h) / 1.5)
            sx = sx if sx >= 0 else 0
            sy = sy if sy >= 0 else 0
            sx = sx if (sx + sw) <= fw else (fw - sw)
            sy = sy if (sy + sh) <= fh else (fh - sh)

            if save:
                cv2.imwrite(os.path.join(save_path, f"{time.time()}.png"),
                            cv2.resize(frame[sy:sy + sh, sx:sx + sw], (SAVE_IMAGE_SIZE, SAVE_IMAGE_SIZE)))
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
            frame = cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), color=(0, 255, 255), thickness=2)

        if save:
            frame = cv2.putText(frame, "press D to stop saving", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                                cv2.LINE_AA)
        else:
            frame = cv2.putText(frame, "press S to save", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                                cv2.LINE_AA)

        cv2.imshow('mygan', frame)

        key = cv2.waitKey(100) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            save = True
        elif key == ord('d'):
            save = False

    video.release()
    cv2.destroyAllWindows()

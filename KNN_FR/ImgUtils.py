import PIL.Image
import numpy as np
import io
import cv2
from dlib import get_frontal_face_detector


faceClassifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"
)
faceDetector = get_frontal_face_detector()
# dont move the xml file if you dont want another spontaneous migraine


def npArrayToImgBytes(array, ext="jpeg"):
    buffer = io.BytesIO()
    PIL.Image.fromarray(array).save(buffer, ext)
    return buffer.getvalue()


def imgBytesToNpArray(imgBytes):
    return np.asarray(PIL.Image.open(io.BytesIO(imgBytes)))


def getFaceFrames(img, border=0.1) -> list:
    if isinstance(img, bytes):
        return getFaceFrames(imgBytesToNpArray(img))
    elif isinstance(img, np.ndarray):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        keyPoints = faceClassifier.detectMultiScale(
            gray, minNeighbors=5, scaleFactor=1.05
        )
        facesArray = []
        for face in keyPoints:
            x, y, w, h = face
            X = int(round(max(x - (w * border), 0)))
            Y = int(round(max(y - (h * border), 0)))
            # (Height, Width, Depth)
            W = int(round(min(w * (1 + border), img.shape[1])))
            H = int(round(min(h * (1 + border), img.shape[0])))
            facesArray.append(
                img[Y: Y + H, X: X + W].copy()
            )
        return facesArray
    else:
        raise RuntimeError("Invalid image data provided")


def getFaceFramesDLIB(img, border=0.1):
    if isinstance(img, bytes):
        return getFaceFramesDLIB(imgBytesToNpArray(img))
    elif isinstance(img, np.ndarray):
        faces = faceDetector(img)
        facesArray = []
        for face in faces:
            w = face.right() - face.left()
            h = face.bottom() - face.top()
            x1 = int(round(max(face.left() - (w*border), 0)))
            x2 = int(round(min(face.right() + (w*border), img.shape[1])))
            y1 = int(round(max(face.top() - (h*border), 0)))
            y2 = int(round(min(face.bottom()+(h*border), img.shape[0])))
            facesArray.append(
                img[y1:y2, x1:x2].copy()
            )
        return facesArray
    else:
        raise RuntimeError("Invalid image data provided")

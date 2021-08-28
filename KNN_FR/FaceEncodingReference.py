from deprecated import deprecated
from .ImgUtils import imgBytesToNpArray
import os
from face_recognition import face_encodings, load_image_file, compare_faces
import re


class FaceEncodingReference:
    def __init__(self, encodings={}):
        self.encodings = encodings

    def addFaceEncoding(self, indentifier: str, encoding):
        if indentifier in self.encodings.keys():
            self.encodings[indentifier].append(encoding)
        else:
            self.encodings[indentifier] = []
            self.addFaceEncoding(indentifier, encoding)

    def loadFromDirectory(self, imgDir: str):
        if not (os.path.isdir(imgDir) and all(
            [os.path.isfile(imgDir + "\\" + path)
             for path in os.listdir(imgDir)]
        )):
            raise FileNotFoundError("You goofed the image directory")
        else:
            for imgName in os.listdir(imgDir):
                name = re.search(r"^\D*", imgName).group()
                img = load_image_file(imgDir + "\\" + imgName)
                encoding = face_encodings(img)[0]
                self.addFaceEncoding(name, encoding)

    def addEncodingFromImageBytes(self, identifier: str, imgBytes: bytes):
        imgArray = imgBytesToNpArray(imgBytes)
        encodings = face_encodings(imgArray)
        if len(encodings) != 1:  # Either no people or two+ people in image
            raise IdentificationError(
                "There can only be one person in reference image")
        self.addFaceEncoding(identifier, encodings[0])

    @deprecated(reason="Use KNN Idendifier instead")
    def identifyCompare(self, encodingToCheck) -> str:
        # this didnt work for very long
        for identifier, encodingList in self.encodings.items():
            comparison = compare_faces(
                encodingList, encodingToCheck, tolerance=0.5)
            if any(comparison):
                return identifier
        raise IdentificationError("Face not in known encodings")

    def clear(self):
        self.encodings.clear()

    def __str__(self) -> str:
        retStr = f"References for {len(self.encodings.keys())} people"
        if len(self.encodings.keys()) <= 5:
            retStr += f":\n"
            for identifier, encodingList in self.encodings.items():
                retStr += f"{identifier}: {len(encodingList)} references available\n"
        return retStr


class IdentificationError(Exception):
    pass

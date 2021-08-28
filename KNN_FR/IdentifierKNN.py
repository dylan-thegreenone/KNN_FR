import math
from face_recognition import face_locations
from .FaceEncodingReference import *
import pickle
import os
from sklearn import neighbors
from .ImgUtils import imgBytesToNpArray
from numpy import ndarray


class IdentifierKNN:
    DEFAULT_DISTANCE_THRESHOLD = 0.6

    def __init__(
        self,
        reference: FaceEncodingReference = None,
        model_save_path=os.path.join(
            os.path.dirname(__file__),
            "model.pkl"
        ),
        reference_save_path=os.path.join(
            os.path.dirname(__file__),
            "reference.pkl"
        )
    ):
        self.reference_save_path = reference_save_path
        self.reference = reference
        if self.reference is None:
            try:
                self.loadReferenceFromFile(self.reference_save_path)
            except Exception:
                self.reference = FaceEncodingReference()
                self.saveReference()
        else:
            self.saveReference()
        self.knn_clf = None
        self.model_save_path = model_save_path
        self.trained = False
        if self.model_save_path is not None:
            try:
                self.loadModelFromFile(self.model_save_path)
            except FileNotFoundError:
                pass

    def loadModelFromFile(self, path):
        with open(path, "rb") as file:
            self.knn_clf = pickle.load(file)
            if not isinstance(self.knn_clf, neighbors.KNeighborsClassifier):
                self.knn_clf = None
                raise Exception("Invalid classifier file")
            else:
                self.trained = True

    def loadReferenceFromFile(self, path):
        with open(path, "rb") as file:
            self.reference = pickle.load(file)
            if not isinstance(self.reference, FaceEncodingReference):
                self.reference = FaceEncodingReference()
                raise Exception("Invalid reference file")

    def saveModel(self):
        if self.model_save_path is not None:
            with open(self.model_save_path, "wb") as file:
                pickle.dump(self.knn_clf, file)

    def saveReference(self):
        if self.reference_save_path is not None:
            with open(self.reference_save_path, "wb") as file:
                pickle.dump(self.reference, file)

    def train(
        self,
        n_neighbors=None,
        knn_algo='ball_tree',
        verbose=False
    ):
        # ha tomorrow me i understand and you dont
        X = []
        y = []
        for identifier, encodings in self.reference.encodings.items():
            for encoding in encodings:
                X.append(encoding)
                y.append(identifier)
        if n_neighbors is None:
            n_neighbors = int(round(math.sqrt(len(X))))
            if verbose:
                print("Chose n_neighbors automatically:", n_neighbors)
        self.knn_clf = neighbors.KNeighborsClassifier(
            n_neighbors=n_neighbors,
            algorithm=knn_algo,
            weights='distance'
        )
        if len(X) > 0 and len(y) > 0:
            self.knn_clf.fit(X, y)
            self.saveModel()
            self.trained = True
        else:
            raise IdentificationError("No encodings available in reference")

    def setReference(self, reference: FaceEncodingReference):
        if not isinstance(reference, FaceEncodingReference):
            raise TypeError(
                "Face reference can only be an instance of FaceEncodingReference"
            )
        self.reference = reference
        self.train()

    def addReferenceImg(self, identifier: str, img: bytes, retrain=False):
        self.reference.addEncodingFromImageBytes(identifier, img)
        if retrain:
            self.train()

    def fullReset(self):
        self.knn_clf = None
        self.reference.clear()
        self.trained = False

    def setSavePath(self, path):
        self.model_save_path = path

    def identify(self, img, distance_threshold=DEFAULT_DISTANCE_THRESHOLD):
        if self.knn_clf is None:
            raise RuntimeError("Classifier not trained yet")
        if isinstance(img, bytes):  # byte image
            return self.identify(imgBytesToNpArray(img))
        elif isinstance(img, ndarray):
            faceLocs = face_locations(img)
            if len(faceLocs) == 1:
                encodings = face_encodings(img)
                closest_distances = self.knn_clf.kneighbors(
                    encodings, n_neighbors=1)
                is_match = closest_distances[0][0][0] <= distance_threshold
                result = {
                    "id": self.knn_clf.predict(encodings)[0] if is_match else "unknown",
                    "face_location": faceLocs[0],
                    "distance_factor": closest_distances[0][0][0]
                }
                # do those zeroes look like eye balls or do i just need to go to sleep
                return result
            else:
                if len(faceLocs) == 0:
                    raise IdentificationError("No faces found in image")
                else:
                    raise IdentificationError(
                        f"Too many faces found in frame: {len(faceLocs)}")

    def __str__(self):
        return self.knn_clf.__str__()

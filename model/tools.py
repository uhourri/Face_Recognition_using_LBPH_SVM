import numpy as np
from skimage import feature
import cv2
import pickle


def sliding_window(image, batch):
    for y in range(0, image.shape[0], batch[0]):
        for x in range(0, image.shape[1], batch[1]):
            yield x, y, image[y:y + batch[1], x:x + batch[0]]


def find_lbp_histogram(image, n_batch=(8, 8)):
    features = []
    batch_size = (
        int(np.floor(image.shape[0] / n_batch[0])),
        int(np.floor(image.shape[1] / n_batch[1]))
    )

    lbp_img = feature.local_binary_pattern(image, P=8, R=1, method="default")

    for (x, y, C) in sliding_window(lbp_img, batch_size):
        if C.shape[0] != batch_size[0] or C.shape[1] != batch_size[1]:
            continue

        H = np.histogram(C, bins=64, density=True)[0]
        H = H.astype("float")
        H /= H.sum()

        features.extend(H)

    return features


def detect_faces(gray_image):
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")
    return faceCascade.detectMultiScale(
        gray_image,
        scaleFactor=1.2,
        minNeighbors=3
    )


def predict(image, model_path):
    with open(model_path, 'rb') as f:
        svm_clf = pickle.load(f)

    threshold = 0.3

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_locations = detect_faces(image)

    if len(face_locations) == 0:
        return []

    faces_encodings = []
    for (x, y, w, h) in face_locations:
        cropped_face = image[y:y + h, x:x + w]
        resized_image = cv2.resize(cropped_face, (128, 128), interpolation=cv2.INTER_AREA)
        encoded_image = find_lbp_histogram(resized_image)
        faces_encodings.append(encoded_image)

    faces = []
    predictions = svm_clf.predict_proba(faces_encodings)
    for i in range(len(face_locations)):
        max_pred = np.argmax(predictions[i])
        if predictions[i][max_pred] > threshold:
            faces.append([svm_clf.classes_[max_pred], face_locations[i]])
        else:
            faces.append(['Unknown', face_locations[i]])

    return faces


def draw_in_image(image_path):
    image = cv2.imread(image_path)
    prediction = predict(image, 'model/Face_Recognition_LBPH_SVM.clf')
    thickness = round(-0.9601 + 0.0031 * image.shape[0] + 0.0008 * image.shape[1])
    for i in range(len(prediction)):
        x, y, w, h = prediction[i][1]
        image = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), thickness)
        cv2.putText(image, prediction[i][0], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    thickness / 2.3 + 0.3, (255, 255, 255), round(thickness * 1.3))

    return image
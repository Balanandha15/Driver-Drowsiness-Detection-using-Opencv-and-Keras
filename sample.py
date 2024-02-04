import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PyQt5.QtCore import Qt, QTimer, QTime, QDate
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel

def detection(cascade, frame, drowsiness_model, scale_factor, color, label):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = cascade.detectMultiScale(frame_gray, scaleFactor=scale_factor, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    
    for (x, y, w, h) in detections:
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        region_of_interest = frame[y:y+h, x:x+w]
        roi_resized = cv2.resize(region_of_interest, (24, 24))
        roi_resized = roi_resized.astype('float32') / 255
        detection_pred = drowsiness_model.predict(np.array([roi_resized]))[0]
        
        if detection_pred > 0.5:
            detection_label = 'Awake'
            text_color = (0, 0, 255)  # Red
        else:
            detection_label = 'Drowsy'
            text_color = (0, 255, 0)  # Green
        
        cv2.putText(frame, label.format(detection_label), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

def preprocess_image(face, img):
    face_rect = cv2.resize(face, (24, 24))
    face_rect = face_rect.astype('float32')
    face_rect /= 255
    return face_rect

class MainApp(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_pretrained_models()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Driver Drowsiness Detection')
        self.setGeometry(1000, 1000, 1000, 1000)

        self.capture = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(5)

        layout = QVBoxLayout()
        self.label = QLabel()
        self.label.setFixedSize(1000, 1000)
        layout.addWidget(self.label)
        self.setLayout(layout)

        # Display date and time
        self.date_time_label = QLabel(self)
        self.date_time_label.setStyleSheet('font-size: 12px; color: #949494;')
        self.date_time_label.setFixedSize(200, 30)
        self.date_time_label.move(500, 50)
        self.date_time_label.show()
        self.update_date_time()

        self.show()

    def update_date_time(self):
        current_time = QTime.currentTime()
        current_date = QDate.currentDate()
        current_time_str = current_time.toString('hh:mm:ss')
        current_date_str = current_date.toString('dd/MM/yyyy')
        self.date_time_label.setText('Date: {} | Time: {} '.format(current_date_str, current_time_str))
        QTimer.singleShot(1000, self.update_date_time)

    def update_frame(self):
        ret, frame = self.capture.read()
        if ret:
            faces, eyes, bodies = self.detect_objects(frame)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                face = frame[y:y+h, x:x+w]
                face_rect = preprocess_image(face, frame)
                face_pred = self.drowsiness_model.predict(np.array([face_rect]))[0]
                if face_pred > 0.5:
                    label = 'Awake'
                    color = (0, 0, 255)
                else:
                    label = 'Drowsy'
                    color = (0, 255, 0)
                cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            for (x, y, w, h) in eyes:
                detection(self.eyes_cascade, frame, self.drowsiness_model, 1.1, (255, 0, 0), 'Eyes: {}')

            for (x, y, w, h) in bodies:
                detection(self.body_cascade, frame, self.drowsiness_model, 1.1, (0, 0, 255), 'Body: {}')

            image = self.convert_cv_qt(frame).toImage()
            self.label.setPixmap(QPixmap.fromImage(image))
            self.label.setAlignment(Qt.AlignCenter)

    def convert_cv_qt(self, frame):
        height, width, channel = frame.shape
        bytesPerLine = 3 * width
        qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
        return QPixmap.fromImage(qImg.rgbSwapped())

    def load_pretrained_models(self):
        eyes_cascade = cv2.CascadeClassifier('C:/Users/balan/M.sc/First year/SET paper/Project/project/haarcascade_eye.xml')
        left_eyes_cascade = cv2.CascadeClassifier('C:/Users/balan/M.sc/First year/SET paper/Project/project/haarcascade_lefteye_2splits.xml')
        right_eyes_cascade = cv2.CascadeClassifier('C:/Users/balan/M.sc/First year/SET paper/Project/project/haarcascade_righteye_2splits.xml')
        drowsiness_model = load_model('InceptionV3_model.h5')
        
        self.face_cascade = left_eyes_cascade
        self.eyes_cascade = eyes_cascade
        self.body_cascade = right_eyes_cascade
        self.drowsiness_model = drowsiness_model

        return left_eyes_cascade, eyes_cascade, right_eyes_cascade, drowsiness_model

    def detect_objects(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(frame_gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        eyes = self.eyes_cascade.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        bodies = self.body_cascade.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        return faces, eyes, bodies

if __name__ == '__main__':
    app = QApplication([])
    main_app = MainApp()
    app.exec_()

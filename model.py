import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PyQt5.QtCore import Qt, QTimer, QTime, QDate
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel

def face_detection(face_cascade, frame, scale_factor):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame_gray, scaleFactor=scale_factor, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    return faces

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
            faces = face_detection(self.face_cascade, frame, 1.3)
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

            image = self.convert_cv_qt(frame).toImage()
            self.label.setPixmap(QPixmap.fromImage(image))
            self.label.setAlignment(Qt.AlignCenter)

    def convert_cv_qt(self, frame):
        height, width, channel = frame.shape
        bytesPerLine = 3 * width
        qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
        return QPixmap.fromImage(qImg.rgbSwapped())

    def load_pretrained_models(self):
        face_cascade = cv2.CascadeClassifier('C:/Users/balan/M.sc/First year/SET paper/Project/project/haarcascade_eye.xml')
        drowsiness_model = load_model('InceptionV3_model.h5')
        self.face_cascade = face_cascade
        self.drowsiness_model = drowsiness_model

if __name__ == '__main__':
    app = QApplication([])
    main_app = MainApp()
    app.exec_()

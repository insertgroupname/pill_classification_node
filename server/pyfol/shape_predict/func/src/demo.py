import sys, os
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout,QPushButton
from PyQt5.QtCore import Qt,pyqtSlot
from PyQt5.QtGui import QPixmap
from func import fed

img_path = ''
# prediction = 'Prediction'


class ImageLabel(QLabel):
    def __init__(self):
        super().__init__()

        self.setAlignment(Qt.AlignCenter)
        self.setText('\n\n Drop Image Here \n\n')
        self.setStyleSheet('''
            QLabel{
                border: 2px 
            }
        ''')

    def setPixmap(self, image):
        super().setPixmap(image)


class AppDemo(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedWidth(400)
        self.setFixedHeight(400)
        self.setAcceptDrops(True)
        button = QPushButton('Predict Shape', self)
        button.clicked.connect(self.on_click)
        # button2 = QPushButton('Predict Color', self)
        # button2.clicked.connect(self.on_click2)
        mainLayout = QVBoxLayout()

        self.photoViewer = ImageLabel()
        mainLayout.addWidget(self.photoViewer)
        mainLayout.addWidget(button)
        self.setLayout(mainLayout)

    @pyqtSlot()
    def on_click(self):
        print(img_path)
        # number_polygon, shape = fed.shapeDetector(img_path)
        # print(number_polygon, shape)
        prediction = fed.colorPrediction(img_path)
        print(prediction)

    # @pyqtSlot()
    # def on_click2(self):
    #     print(img_path)
    #     prediction = fed.colorPrediction(img_path)
    #     print(prediction)

    def dragEnterEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        global img_path
        if event.mimeData().hasImage:
            event.setDropAction(Qt.CopyAction)
            file_path = event.mimeData().urls()[0].toLocalFile()
            self.set_image(file_path)
            img_path = file_path
            event.accept()
        else:
            event.ignore()

    def set_image(self, file_path):
        self.photoViewer.setPixmap(QPixmap(file_path))


app = QApplication(sys.argv)
demo = AppDemo()
demo.show()
sys.exit(app.exec_())
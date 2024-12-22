import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt5.uic import loadUi
from PIL import Image
import HW2_Q2
from modelGD import Discriminator, Generator
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class Main_UI(QMainWindow):
    def __init__(self, parent = None):
        super(Main_UI,self).__init__(parent)
        loadUi("./HW2.ui",self)
        self.device=None
        self.Gmodel=None
        self.Dmodel=None
        self.transform=None
        self.Data_loader=None
        self.n_Data_loader=None
        self.files=[]
        self.loss="./loss_ver3.png"
        self.loadFiles=r"D:\CPJ\courses\1131\CvDL\Q2_images\mnist"
        self.result_img="./real_fake_ver3.png"
        HW2_Q2.initial(self)
        self.Connect_btn()


    def Connect_btn(self):
        self.pushButton.clicked.connect(self.showMessage_wrong_btn)           
        self.pushButton_2.clicked.connect(self.showMessage_wrong_btn)         
        self.pushButton_3.clicked.connect(self.showMessage_wrong_btn)         
        self.pushButton_4.clicked.connect(self.showMessage_wrong_btn)         
        self.pushButton_5.clicked.connect(self.showMessage_wrong_btn)         
        self.pushButton_6.clicked.connect(self.pushButton6F)        
        self.pushButton_7.clicked.connect(self.pushButton7F)        
        self.pushButton_8.clicked.connect(self.pushButton8F)        
        self.pushButton_9.clicked.connect(self.pushButton9F)   

    
    def showMessage_wrong_btn(self):
        msg = QMessageBox()
        msg.setWindowTitle("操作錯誤提示")
        msg.setText("非此題按鈕")
        msg.setIcon(QMessageBox.Information)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def pushButton6F(self):
        HW2_Q2.show_augmentation(self)
    def pushButton7F(self):
        HW2_Q2.show_structure(self.Gmodel,self.Dmodel)
    def pushButton8F(self):
        HW2_Q2.show_loss(self.loss)
    def pushButton9F(self):
        HW2_Q2.show_product_image(self)
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Main_UI()
    window.show()
    sys.exit(app.exec_())
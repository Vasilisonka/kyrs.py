import sys  # sys нужен для передачи argv в QApplication
from PyQt5 import QtWidgets
import hi_window
import new_patient
import spravka
import delete

class Win1(QtWidgets.QMainWindow): # pyuic5 hi_window.ui -o hi_window.py
    def __init__(self):
        super(Win1, self).__init__()
        self.ui = hi_window.Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.combobox_Patient.addItem('Донор')
        self.ui.combobox_Patient.addItem('Реципиент')
        self.ui.comboBox_group.addItem('I')
        self.ui.comboBox_group.addItem('II')
        self.ui.comboBox_group.addItem('III')
        self.ui.comboBox_group.addItem('IV')
        self.ui.comboBox_rezus.addItem('Положительный')
        self.ui.comboBox_rezus.addItem('Отрицательный')
        self.ui.Add.clicked.connect(self.show_Win2)
        self.ui.action.triggered.connect(self.show_Win3)
        self.ui.Delete.clicked.connect(self.show_Win4)
    def show_Win2(self):
        self.ui2 = Win2()
        self.ui2.show()
    def show_Win3(self):
        self.ui3 = Win3()
        self.ui3.show()
    def show_Win4(self):
        self.ui4 = Win4()
        self.ui4.show()


class Win2(QtWidgets.QMainWindow):# pyuic5 new_patient.ui -o new_patient.py
    def __init__(self):
        super(Win2, self).__init__()
        self.ui2 = new_patient.Ui_MainWindow()
        self.ui2.setupUi(self)
        self.ui2.combobox_Patient.addItem('Донор')
        self.ui2.combobox_Patient.addItem('Реципиент')
        self.ui2.combobox_group.addItem('I')
        self.ui2.combobox_group.addItem('II')
        self.ui2.combobox_group.addItem('III')
        self.ui2.combobox_group.addItem('IV')
        self.ui2.combobox_rezus.addItem('Положительный')
        self.ui2.combobox_rezus.addItem('Отрицательный')
        self.ui2.combobox_pol.addItem('женский')
        self.ui2.combobox_pol.addItem('мужской')

class Win3(QtWidgets.QMainWindow): # pyuic5 spravka.ui -o spravka.py
    def __init__(self):
        super(Win3, self).__init__()
        self.ui3 = spravka.Ui_MainWindow()
        self.ui3.setupUi(self)
class Win4(QtWidgets.QMainWindow): # pyuic5 delete.ui -o delete.py
    def __init__(self):
        super(Win4, self).__init__()
        self.ui4 = delete.Ui_Delete()
        self.ui4.setupUi(self)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Win1()
    window.show()
    app.exec_()
import sys  
from PyQt5.QtWidgets import QMainWindow
from .src import hi_window as hi_window
from .src import new_patient as new_patient
from .src import spravka as spravka
from .src import delete as delete

class MainWindow(QMainWindow): # pyuic5 hi_window.ui -o hi_window.py
    def __init__(self, controller):
        super(MainWindow, self).__init__()
        self._main_controller = controller
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
        self.ui2.show()
    def show_Win3(self):
        self.ui3 = InfoWindow()
        self.ui3.show()
    def show_Win4(self):
        self.ui4 = DeleteWindow(self._main_controller)
        self.ui4.show()


class AddWindow(QMainWindow):# pyuic5 new_patient.ui -o new_patient.py
    def __init__(self, controller):
        super().__init__()
        self.ui2 = new_patient.Ui_MainWindow()
        self.ui2.setupUi(self)
        self._main_controller = controller
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

        self.ui2.add.clicked.connect(self._main_controller.add_patient)

class InfoWindow(QMainWindow): # pyuic5 spravka.ui -o spravka.py
    def __init__(self):
        super(InfoWindow, self).__init__()
        self.ui3 = spravka.Ui_MainWindow()
        self.ui3.setupUi(self)
class DeleteWindow(QMainWindow): # pyuic5 delete.ui -o delete.py
    def __init__(self, controller):
        super(DeleteWindow, self).__init__()
        self.ui4 = delete.Ui_Delete()
        self.ui4.setupUi(self)
        self._main_controller = controller
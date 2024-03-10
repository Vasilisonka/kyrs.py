import sys
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QMainWindow, QTableWidgetItem
from .src import hi_window as hi_window
from .src import new_patient as new_patient
from .src import spravka as spravka
from .src import delete as delete

class MainWindow(QMainWindow): # pyuic5 hi_window.ui -o hi_window.py
    def __init__(self, controller):
        super(MainWindow, self).__init__()

        self._main_controller = controller
        self.header_row = ["№", "Имя", "Фамилия", "Отчество", "Тип", "Группа крови", "Rh"]

        self.ui = hi_window.Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.combobox_Patient.addItem('Донор')
        self.ui.combobox_Patient.addItem('Реципиент')
        self.ui.combobox_Patient.currentIndexChanged.connect(self.on_combobox_Patient_changed)

        self.ui.comboBox_group.addItem('I')
        self.ui.comboBox_group.addItem('II')
        self.ui.comboBox_group.addItem('III')
        self.ui.comboBox_group.addItem('IV')
        self.ui.comboBox_group.currentIndexChanged.connect(self.on_comboBox_group_changed)

        self.ui.comboBox_rezus.addItem('Положительный')
        self.ui.comboBox_rezus.addItem('Отрицательный')
        self.ui.comboBox_rezus.currentIndexChanged.connect(self.on_comboBox_rezus_changed)

        self.ui.Add.clicked.connect(self.show_Win2)
        self.ui.action.triggered.connect(self.show_Win3)
        self.ui.Delete.clicked.connect(self._main_controller.delete_row)

        self.ui.lineEdit.textChanged.connect(self.on_surname_change)
        self.ui.lineEdit_2.textChanged.connect(self.on_name_change)
        self.ui.lineEdit_3.textChanged.connect(self.on_patronymic_change)

    @pyqtSlot()
    def on_surname_change(self):
        self.filter_table(1, self.ui.lineEdit.text())

    @pyqtSlot()
    def on_name_change(self):
        self.filter_table(2, self.ui.lineEdit_2.text())

    @pyqtSlot()
    def on_patronymic_change(self):
        self.filter_table(3, self.ui.lineEdit_3.text())

    @pyqtSlot()
    def on_combobox_Patient_changed(self):
        self.filter_table(4, self.ui.combobox_Patient.currentText())

    @pyqtSlot()
    def on_comboBox_group_changed(self):
        self.filter_table(5, self.ui.comboBox_group.currentText())

    @pyqtSlot()
    def on_comboBox_rezus_changed(self):
        self.filter_table(6, self.ui.comboBox_rezus.currentText())

    def filter_table(self, column_ind, text):
        search_text = text.lower()
        for row_ind in range(self.ui.tableWidget.rowCount()):
            item = self.ui.tableWidget.item(row_ind, column_ind)

            if item is not None:
                cell_text = item.text().lower()
                row_visible = search_text in cell_text
                self.ui.tableWidget.setRowHidden(row_ind, not row_visible)
        self.ui.tableWidget.setColumnHidden(0, True)
    
    def arabic_to_roman(self, num):
        match num:
            case 0:
                return "I"
            case 1:
                return "II"
            case 2: 
                return "III"
            case 3:
                return "IV"
    
    def fill_table(self, data):

        self.ui.tableWidget.clearContents()
        self.ui.tableWidget.setRowCount(len(data))
        self.ui.tableWidget.setColumnCount(len(data[0]))
        self.ui.tableWidget.setHorizontalHeaderLabels(self.header_row)

        for row_num, row_data in enumerate(data):
            for col_num, col_data in enumerate(row_data):
                match col_num:
                    case 4:
                        item = QTableWidgetItem(str('Донор' if int(col_data) == 0 else 'Реципиент'))
                    case 5:
                        item = QTableWidgetItem(self.arabic_to_roman(col_data))
                    case 6:
                        item = QTableWidgetItem(str('Положительный' if int(col_data) == 0 else 'Отрицательный'))
                    case _:
                        item = QTableWidgetItem(str(col_data))
                
                self.ui.tableWidget.setItem(row_num, col_num, item)
        self.ui.tableWidget.setColumnHidden(0, True)

    def show_Win2(self):
        self.ui2.show()
    def show_Win3(self):
        self.ui3 = InfoWindow()
        self.ui3.show()
    # def show_Win4(self):
    #     self.ui4 = DeleteWindow(self._main_controller)
    #     self.ui4.show()


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
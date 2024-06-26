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
        self.header_row = ["№", "Имя", "Фамилия", "Отчество", "Группа крови", "Пол", "Возраст"]
        self.header_row_donors = ["№", "Имя", "Группа крови", "Пол"]

        self.blood_type_relation = {
            "O-":"O-",
            "O+":"O-O+",
            "A-":"O-A-",
            "A+":"O-O+A-A+",
            "B-":"O-B-",
            "B+":"O-O+B-B+",
            "AB-":"O-A-B-AB-",
            "AB+":"O-O+A-A+B-B+AB-AB+"
        }

        self.ui = hi_window.Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.comboBox_group.addItem('O-')
        self.ui.comboBox_group.addItem('O+')
        self.ui.comboBox_group.addItem('A-')
        self.ui.comboBox_group.addItem('A+')
        self.ui.comboBox_group.addItem('B-')
        self.ui.comboBox_group.addItem('B+')
        self.ui.comboBox_group.addItem('AB-')
        self.ui.comboBox_group.addItem('AB+')
        self.ui.comboBox_group.currentIndexChanged.connect(self.on_comboBox_group_changed)

        self.ui.Add.clicked.connect(self.show_Win2)
        self.ui.action.triggered.connect(self.show_Win3)
        self.ui.Delete.clicked.connect(self._main_controller.delete_row)

        self.ui.lineEdit.textChanged.connect(self.on_surname_change)
        self.ui.lineEdit_2.textChanged.connect(self.on_name_change)
        self.ui.lineEdit_3.textChanged.connect(self.on_patronymic_change)

        self.ui.tableWidget.itemClicked.connect(self.filter_donor_table)

    @pyqtSlot()
    def on_surname_change(self):
        self.filter_table(1, self.ui.lineEdit.text())

    @pyqtSlot()
    def on_name_change(self):
        self.filter_table(1, self.ui.lineEdit_2.text())

    @pyqtSlot()
    def on_patronymic_change(self):
        self.filter_table(1, self.ui.lineEdit_3.text())

    @pyqtSlot()
    def on_comboBox_group_changed(self):
        self.filter_table(2, self.ui.comboBox_group.currentText())
    
    @pyqtSlot()
    def on_table_wiget_itemClicked(self, item):
        self.filter_donor_table(item)

    def filter_donor_table(self, item):
        current_row = self.ui.tableWidget.currentRow()
        blood_type = self.ui.tableWidget.item(current_row, 4).text()

        for row_ind in range(self.ui.tableWidget_2.rowCount()):
            item = self.ui.tableWidget_2.item(row_ind, 2)
            if item is not None:
                cell_text = item.text()
                row_visible = cell_text in self.blood_type_relation[blood_type]
                self.ui.tableWidget_2.setRowHidden(row_ind, not row_visible)
        self.ui.tableWidget_2.setColumnHidden(0, True)
        

    def filter_table(self, column_ind, text):
        search_text = text.lower()
        for row_ind in range(self.ui.tableWidget_2.rowCount()):
            item = self.ui.tableWidget_2.item(row_ind, column_ind)

            if item is not None:
                cell_text = item.text().lower()
                row_visible = search_text == cell_text
                self.ui.tableWidget_2.setRowHidden(row_ind, not row_visible)
        self.ui.tableWidget_2.setColumnHidden(0, True)
            
    def fill_donor_table(self, data):
        self.ui.tableWidget_2.clearContents()
        self.ui.tableWidget_2.setRowCount(len(data))
        self.ui.tableWidget_2.setColumnCount(len(data[0]))
        self.ui.tableWidget_2.setHorizontalHeaderLabels(self.header_row_donors)

        for row_num, row_data in enumerate(data):
            for col_num, col_data in enumerate(row_data):
                match col_data:
                    case _:
                        item = QTableWidgetItem(str(col_data))
                self.ui.tableWidget_2.setItem(row_num, col_num, item)
        self.ui.tableWidget_2.setColumnHidden(0, True)
    
    def fill_table(self, data):
        self.ui.tableWidget.clearContents()
        self.ui.tableWidget.setRowCount(len(data))
        self.ui.tableWidget.setColumnCount(len(data[0]))
        self.ui.tableWidget.setHorizontalHeaderLabels(self.header_row)

        for row_num, row_data in enumerate(data):
            for col_num, col_data in enumerate(row_data):
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
        # self.ui2.combobox_Patient.addItem('Донор')
        # self.ui2.combobox_Patient.addItem('Реципиент')
        self.ui2.combobox_group.addItem('O-')
        self.ui2.combobox_group.addItem('O+')
        self.ui2.combobox_group.addItem('A-')
        self.ui2.combobox_group.addItem('A+')
        self.ui2.combobox_group.addItem('B-')
        self.ui2.combobox_group.addItem('B+')
        self.ui2.combobox_group.addItem('AB-')
        self.ui2.combobox_group.addItem('AB+')
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
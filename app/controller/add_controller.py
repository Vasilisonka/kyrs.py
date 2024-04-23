from PyQt5.QtCore import QObject, pyqtSlot
from ..model.sqlite_model import ModelSQLite
from ..view.sql_view import AddWindow

class ControlerAdd(QObject):
    def __init__(self, model:ModelSQLite, view:AddWindow) -> None:
        super().__init__()

        self.model = model
        self.view = view

    @pyqtSlot()
    def add_patient(self):
        self.model.insert(
            self.view.ui2.lineEdit_2.text(),
            self.view.ui2.lineEdit.text(),
            self.view.ui2.lineEdit_3.text(),
            # self.view.ui2.combobox_Patient.currentIndex(),
            self.view.ui2.combobox_group.currentIndex(),
            # self.view.ui2.combobox_rezus.currentIndex()
        )
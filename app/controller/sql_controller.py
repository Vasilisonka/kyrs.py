from PyQt5.QtCore import pyqtSlot
from ..model.sqlite_model import ModelSQLite
from ..view.sql_view import MainWindow, AddWindow, DeleteWindow

class ControlerSQLite():
    def __init__(self, model:ModelSQLite) -> None:
        self.model = model

    @pyqtSlot(str)
    def add_patient(self, name:str, surname:str, patronymic:str, type:str, blood_type:str, rh:str):
        self.model.insert(name, surname, patronymic, type, blood_type, rh)
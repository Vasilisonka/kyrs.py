from PyQt5.QtCore import QObject, pyqtSignal
from .src import sqlite_backend as sql

class ModelSQLite(QObject):
    data_changed = pyqtSignal()
    def __init__(self):
        super().__init__()
        self._connection = sql.connect_to_db(sql.DB_name)
        sql.create_table(self._connection)
    
    @property
    def connection(self):
        return self._connection
    
    def insert(self, name:str, surname:str, patronymic:str, type:str, blood_type:str, rh:str):
        sql.insert(self._connection, name, surname, patronymic, type, blood_type, rh)
        self.data_changed.emit()
    
    def delete_object(self, id):
        sql.delete(self._connection, id)
        self.data_changed.emit()
    
    def read_all(self):
        return sql.read_all(self._connection)
    
    def read_donor_data(self):
        return sql.read_donor_data(self._connection)
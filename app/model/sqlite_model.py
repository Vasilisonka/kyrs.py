from .src import sqlite_backend as sql

class ModelSQLite(object):
    
    def __init__(self):
        self._connection = sql.connect_to_db(sql.DB_name)
        sql.create_table(self._connection)
    
    @property
    def connection(self):
        return self._connection
    
    def insert(self, name:str, surname:str, patronymic:str, type:str, blood_type:str, rh:str):
        sql.insert(self._connection, name, surname, patronymic, type, blood_type, rh)
    
    def delete(self, id):
        sql.delete(self._connection, id)
    
    def read_all(self):
        return sql.read_all(self._connection)
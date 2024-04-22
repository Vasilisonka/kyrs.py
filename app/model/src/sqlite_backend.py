import sqlite3
from sqlite3 import OperationalError, IntegrityError, ProgrammingError

DB_name = "Users.db"

def connect_to_db(db=None):
    connection = sqlite3.connect(db)
    return connection

def connect(func):
    def inner_func(connection:sqlite3.Connection, *args, **kwargs):
        cursor = connection.cursor()
        try:
            cursor.execute(
                'SELECT name FROM sqlite_master WHERE type="table";')
        except(AttributeError, ProgrammingError):
            connection = connect_to_db(DB_name)
        return func(connection, *args, *kwargs)
    return inner_func

def disconect_from_db(db=None, connection:sqlite3.Connection=None):
    if connection is not None:
        connection.close()

@connect
def create_table(connection:sqlite3.Connection):
    cursor = connection.cursor()
    try:
        cursor.execute('SELECT name FROM sqlite_master WHERE type="table" AND name="USERS"')
    except OperationalError:
        sql = '''
            CREATE TABLE USERS (user_id INTEGER PRIMARY KEY,
            name TEXT,
            surname TEXT,
            patronymic TEXT,
            type TEXT,
            blood_type INTEGER,
            rh TEXT);
            '''
        cursor.execute(sql)
        connection.commit()

@connect
def insert(connection:sqlite3.Connection, name:str, surname:str, patronymic:str, type:str, blood_type:str, rh:str):
    cursor = connection.cursor()
    sql = '''INSERT INTO USERS (
        name,
        surname,
        patronymic,
        type,
        blood_type,
        rh
    ) VALUES (?, ?, ?, ?, ?, ?);'''

    try:
        cursor.execute(sql, (name, surname, patronymic, type, blood_type, rh))
        connection.commit()
    except IntegrityError as e:
        print(e)

@connect
def read_all(connection:sqlite3.Connection) -> list:
    cursor = connection.cursor()
    sql = '''SELECT * FROM USERS;'''
    c = cursor.execute(sql)
    result = c.fetchall()
    return result

@connect
def read_donor_data(connection:sqlite3.Connection) -> list:
    cursor = connection.cursor()
    sql = '''SELECT * FROM Donors;'''
    c = cursor.execute(sql)
    result = c.fetchall()
    return result

@connect
def delete(connection:sqlite3.Connection, id):
    cursor = connection.cursor()
    sql_check = '''SELECT EXISTS(SELECT * FROM USERS WHERE user_id=?);'''
    sql_delete = '''DELETE FROM USERS WHERE user_id=?;'''
    c = cursor.execute(sql_check, (id,))
    result = c.fetchone()
    if result[0]:
        c.execute(sql_delete, (id,))
        connection.commit()
    else:
        print()
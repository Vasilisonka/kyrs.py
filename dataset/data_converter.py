import pandas as pd
import csv
import sqlite3 as sql

sql_connection = sql.connect('Users.db')
cursor = sql_connection.cursor()

with open('dataset/healthcare_dataset.csv', 'r') as fin:
    dr = csv.DictReader(fin)
    to_db = [(item['Name'], item['Blood Type']) for item in dr]

cursor.executemany('INSERT INTO Donors (Name, BloodType) VALUES (?, ?);', to_db)
sql_connection.commit()
sql_connection.close()

import csv
import sqlite3 as sql

sql_connection = sql.connect('Users.db')
cursor = sql_connection.cursor()

with open('dataset/healthcare_dataset.csv', 'r') as fin:
    dr = csv.DictReader(fin)
    to_db = [(item['Name'], item['Blood Type'], item['Gender']) for item in dr]

cursor.executemany('INSERT INTO Donors (Name, bloodType, Sex) VALUES (?, ?, ?);', to_db)
sql_connection.commit()
sql_connection.close()

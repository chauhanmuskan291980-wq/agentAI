import sqlite3

# Connect to SQLite database
connection = sqlite3.connect("student.db")

# Create cursor
cursor = connection.cursor()

# Create table
table_info = """
CREATE TABLE IF NOT EXISTS STUDENT (
    NAME VARCHAR(25),
    CLASS VARCHAR(25),
    SECTION VARCHAR(25),
    MARKS INT
)
"""

cursor.execute(table_info)

# Insert records
cursor.execute("INSERT INTO STUDENT VALUES ('Krish','Data Science','A',90)")
cursor.execute("INSERT INTO STUDENT VALUES ('Muskan','AI Engineering','A',95)")
cursor.execute("INSERT INTO STUDENT VALUES ('Ishaan','Football','B',78)")
cursor.execute("INSERT INTO STUDENT VALUES ('Aravind Srinivas','Artificial Intelligence','A',99)")
cursor.execute("INSERT INTO STUDENT VALUES ('Priya','Computer Science','C',88)")
cursor.execute("INSERT INTO STUDENT VALUES ('Rahul','Mechanical Engineering','B',82)")
cursor.execute("INSERT INTO STUDENT VALUES ('Sneha','Electronics','A',91)")
cursor.execute("INSERT INTO STUDENT VALUES ('Rohan','Cyber Security','C',85)")



print("Data inserted successfully!")
data = cursor.execute(''' Select * from STUDENT''')

for row in data:
    print(row)


## Commit your change in the databse
connection.commit()
connection.close()
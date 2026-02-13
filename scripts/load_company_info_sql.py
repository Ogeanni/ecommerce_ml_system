import sqlite3

conn = sqlite3.connect("data/processed/company.db")

with open("data/sql/schema.sql") as f:
    conn.executescript(f.read())

with open("data/sql/seed.sql") as f:
    conn.executescript(f.read())

conn.commit()
conn.close()

print(" SQL database created")

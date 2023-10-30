import sqlite3

DATABASE = 'tracker.db'

def create_table():
    with sqlite3.connect(DATABASE) as con:
        cursor = con.cursor()
        
        cursor.execute("""
        CREATE TABLE visited_urls (
            id INTEGER PRIMARY KEY,
            url TEXT
        )
        """)
        con.commit()

    print("Table visited_urls created.")

if __name__ == "__main__":
    create_table()

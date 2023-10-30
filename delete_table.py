import sqlite3

DATABASE = 'tracker.db'

def delete_table():
    with sqlite3.connect(DATABASE) as con:
        cursor = con.cursor()
        
        cursor.execute("DROP TABLE IF EXISTS visited_urls")
        con.commit()

    print("Table visited_urls deleted.")

if __name__ == "__main__":
    delete_table()

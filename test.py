import sqlite3

DATABASE = 'tracker.db'

def get_all_urls():
    with sqlite3.connect(DATABASE) as con:
        cursor = con.cursor()
        cursor.execute("SELECT url FROM visited_urls")
        urls = cursor.fetchall()
    return [url[0] for url in urls]

def main():
    urls = get_all_urls()
    if urls: 
        print("Visited URLs:")
        for idx, url in enumerate(urls, start=1):
            print(f"{idx}. {url}")
    else:
        print("No URLs found in the database.")

if __name__ == "__main__":
    main()

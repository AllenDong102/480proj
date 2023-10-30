from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3

app = Flask(__name__)
CORS(app)
DATABASE = 'tracker.db'

@app.route('/track', methods=['POST'])
def track_url():
    data = request.json
    url = data.get('url')
    with sqlite3.connect(DATABASE) as con:
        cursor = con.cursor()
        cursor.execute("INSERT INTO visited_urls (url) VALUES (?)", (url,))
        con.commit()
    return jsonify({"message": "URL stored successfully!"})

if __name__ == "__main__":
    app.run(debug=True)

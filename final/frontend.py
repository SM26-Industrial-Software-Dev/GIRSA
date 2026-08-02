import sqlite3
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import find_links and DB_PATH from your backend script
from main import DB_PATH, find_links

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/analyze")
async def analyze(data: dict):
    transcript = data.get("transcript", "")

    # Open database connection and run find_links
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        soup_output = find_links(transcript, db_cursor=cursor)

    # Convert the BeautifulSoup object to an HTML string
    return {"message": "Success!", "html": str(soup_output)}

# MOAI 2026 - HTML + Python Web App

This project is a Flask website that recommends songs based on mood.

## Added recommendation stack

- TF-IDF vectors over `lyrics + artist + track + genre`
- Similarity search with cosine metric via FAISS (`faiss-cpu`)
- Automatic fallback to sklearn if FAISS is unavailable
- Prompt template moved to JSON: `prompts/mood_analysis.json`
- UI sections: similar tracks, album link, lyrics link, debug panel

## Genius lyrics database (separate file)

Lyrics are stored in a separate local database:

- `lyrics_database.csv`

Build/update it from Genius:

```powershell
python build_lyrics_database.py --limit 50
```

If Genius blocks the request (common in scripted traffic), the pipeline falls back to `lyrics.ovh`
to keep real lyrics coverage high while preserving Genius search links in UI.

Useful options:

- `--limit 0` scrape all missing tracks
- `--force` rescrape already saved tracks
- `--sleep 1.0` delay between requests

The app auto-loads `lyrics_database.csv` and uses it for vector similarity.

## Run locally (Windows PowerShell)

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Start the site:

```powershell
python app.py
```

4. Open:

```text
http://localhost:8000
```

## Optional API setup

- The app works without API key (local fallback mood analysis).
- For OpenAI mode, set env var before running:

```powershell
$env:API_KEY="your_openai_key_here"
python app.py
```

### Spotify links (strict Spotify resolver)

Set Spotify credentials so track and album links resolve directly from Spotify Web API:

```powershell
$env:SPOTIFY_CLIENT_ID="your_spotify_client_id"
$env:SPOTIFY_CLIENT_SECRET="your_spotify_client_secret"
python app.py
```

Without these variables, the app still keeps Spotify-only links but may fall back to Spotify search URLs.

## One-click start (Windows)

Use the included batch files:

- `start_site.bat` - creates `.venv` if missing, installs requirements, opens browser, starts server.
- `stop_site.bat` - stops running Python server processes.

You can run them by double-clicking in File Explorer or from PowerShell:

```powershell
.\start_site.bat
```

```powershell
.\stop_site.bat
```

## Deploy online

You can deploy this to any Flask-friendly platform (Render, Railway, Fly.io, etc.).

from flask import Flask, render_template, request
import pandas as pd
from openai import OpenAI
import json
import urllib.parse
import os
import re
import requests
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from lyrics_scraper import load_lyrics_database, song_key

try:
    import faiss
except ImportError:
    faiss = None

OPENAI_KEY = None
OPENAI_KEY = os.getenv("API_KEY")

FILENAME_CSV = "baza_piosenek.csv"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, FILENAME_CSV)
LYRICS_DB_FILENAME = "lyrics_database.csv"
LYRICS_DB_PATH = os.path.join(BASE_DIR, LYRICS_DB_FILENAME)
PROMPTS_PATH = os.path.join(BASE_DIR, "prompts", "mood_analysis.json")

GENRE_CATEGORIES = {
    "Wszystkie / Dowolny": [],
    "Rap / Hip-Hop / Drill": ["rap", "hip hop", "hip-hop", "drill", "trap", "baddie", "gangsta", "old school"],
    "Pop / K-Pop": ["pop", "dance", "k-pop", "kpop", "korean", "mainstream"],
    "Rock / Metal / Alternatywa": ["rock", "metal", "punk", "grunge", "indie", "alternative"],
    "R&B / Soul": ["r&b", "rnb", "soul", "blues", "jazz", "chill"],
    "Elektroniczna / Club": ["house", "techno", "edm", "club", "electronic"]
}

app = Flask(__name__)

_SIMILARITY_CACHE = {
    "source_mtime": None,
    "lyrics_mtime": None,
    "engine": None,
}
_LINK_CACHE = {}
_SPOTIFY_TOKEN_CACHE = {"token": None, "expires_at": 0.0}

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "").strip()
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "").strip()


@app.after_request
def add_no_cache_headers(response):
    """Force fresh page state so questionnaire does not persist between app runs."""
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


def load_prompts():
    default_prompt = (
        "Jestes profesjonalnym DJ-em.\\n"
        "Opis nastroju uzytkownika: \"{user_mood}\"\\n\\n"
        "Okresl dwa parametry muzyczne (0.0 do 1.0):\\n"
        "1. Valence (Radosc/Pozytywnosc): 0.0 to smutek/mrok, 1.0 to euforia/szczescie.\\n"
        "2. Energy (Energia): 0.0 to sennosc/spokoj, 1.0 to chaos/szybkosc.\\n\\n"
        "Zwroc TYLKO JSON: {{\"valence\": <float>, \"energy\": <float>, \"diagnosis\": \"<krotki opis w 3 slowach>\"}}"
    )

    if not os.path.exists(PROMPTS_PATH):
        return {"mood_analysis_user": default_prompt}

    try:
        with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if "mood_analysis_user" not in payload:
            payload["mood_analysis_user"] = default_prompt
        return payload
    except Exception:
        return {"mood_analysis_user": default_prompt}


PROMPTS = load_prompts()


def load_song_data():
    try:
        return pd.read_csv(FILE_PATH, on_bad_lines="skip")
    except Exception:
        return pd.DataFrame()


def spotify_track_link(artist, track, csv_link=""):
    safe_csv_link = str(csv_link or "").strip()
    if safe_csv_link.startswith("https://open.spotify.com/track/"):
        return safe_csv_link
    query = urllib.parse.quote(f"{artist} {track}")
    return f"https://open.spotify.com/search/{query}"


def spotify_album_link(artist):
    query = urllib.parse.quote(f"{artist} album")
    return f"https://open.spotify.com/search/{query}"


def genius_lyrics_link(artist, track):
    query = urllib.parse.quote(f"{artist} {track} lyrics")
    return f"https://genius.com/search?q={query}"


def genius_direct_song_link(artist, track):
    """Builds direct Genius song URL slug for a concrete lyrics page."""
    raw = f"{artist}-{track}".lower()
    raw = raw.replace("&", "and")
    raw = re.sub(r"[^a-z0-9]+", "-", raw)
    raw = re.sub(r"-+", "-", raw).strip("-")
    if not raw:
        return genius_lyrics_link(artist, track)
    return f"https://genius.com/{raw}-lyrics"


def _token_overlap_score(artist, track, cand_artist, cand_track):
    def norm_tokens(value):
        tokenized = re.sub(r"[^a-z0-9 ]+", " ", str(value or "").lower())
        return {t for t in tokenized.split() if t}

    a = norm_tokens(artist)
    t = norm_tokens(track)
    ca = norm_tokens(cand_artist)
    ct = norm_tokens(cand_track)

    score = 0.0
    if a:
        score += 2.0 * (len(a.intersection(ca)) / len(a))
    if t:
        score += 2.5 * (len(t.intersection(ct)) / len(t))
    return score


def get_spotify_access_token():
    """Gets app access token for Spotify Web API (Client Credentials)."""
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
        return None

    now = float(pd.Timestamp.utcnow().timestamp())
    cached_token = _SPOTIFY_TOKEN_CACHE.get("token")
    cached_expiry = float(_SPOTIFY_TOKEN_CACHE.get("expires_at", 0.0))
    if cached_token and now < cached_expiry:
        return cached_token

    try:
        response = requests.post(
            "https://accounts.spotify.com/api/token",
            data={"grant_type": "client_credentials"},
            auth=(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET),
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
        token = str(payload.get("access_token", "")).strip()
        expires_in = int(payload.get("expires_in", 3600))
        if not token:
            return None
        _SPOTIFY_TOKEN_CACHE["token"] = token
        _SPOTIFY_TOKEN_CACHE["expires_at"] = now + max(60, expires_in - 60)
        return token
    except Exception:
        return None


def fetch_spotify_links(artist, track, csv_track_link=""):
    """Returns direct Spotify track and album URLs using Spotify Web API."""
    cache_key = f"spotify::{artist.lower().strip()}::{track.lower().strip()}::{str(csv_track_link).lower().strip()}"
    if cache_key in _LINK_CACHE:
        return _LINK_CACHE[cache_key]

    csv_link = str(csv_track_link or "").strip()
    default_result = {
        "track_link": spotify_track_link(artist, track, csv_link=csv_link),
        "album_link": spotify_album_link(artist),
        "resolved": False,
    }

    # If CSV already provides concrete Spotify track URL, use it and infer album by search.
    if csv_link.startswith("https://open.spotify.com/track/"):
        _LINK_CACHE[cache_key] = default_result
        return default_result

    token = get_spotify_access_token()
    if not token:
        _LINK_CACHE[cache_key] = default_result
        return default_result

    query = urllib.parse.quote(f"track:{track} artist:{artist}")
    url = f"https://api.spotify.com/v1/search?q={query}&type=track&limit=15"

    try:
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
        results = payload.get("tracks", {}).get("items", [])
        if not results:
            _LINK_CACHE[cache_key] = default_result
            return default_result

        best = None
        best_score = -1.0
        for item in results:
            artists = item.get("artists", []) or []
            cand_artist = " ".join([str(a.get("name", "")) for a in artists])
            cand_track = str(item.get("name", ""))
            score = _token_overlap_score(artist, track, cand_artist, cand_track)
            if score > best_score:
                best_score = score
                best = item

        if not best:
            _LINK_CACHE[cache_key] = default_result
            return default_result

        track_link = str(best.get("external_urls", {}).get("spotify", "")).strip() or default_result["track_link"]
        album_link = str(best.get("album", {}).get("external_urls", {}).get("spotify", "")).strip() or default_result["album_link"]
        out = {
            "track_link": track_link,
            "album_link": album_link,
            "resolved": True,
        }
        _LINK_CACHE[cache_key] = out
        return out
    except Exception:
        _LINK_CACHE[cache_key] = default_result
        return default_result


def build_lyrics_lookup(lyrics_df):
    lookup = {}
    if lyrics_df is None or lyrics_df.empty:
        return lookup
    for _, row in lyrics_df.iterrows():
        key = str(row.get("song_key", "")).strip()
        if not key:
            continue
        lookup[key] = {
            "genius_url": str(row.get("genius_url", "")).strip(),
            "status": str(row.get("status", "")).strip(),
        }
    return lookup


def resolve_song_links(artist, track, csv_track_link="", lyrics_entry=None):
    """Resolve concrete links for play, album and lyrics actions."""
    spotify = fetch_spotify_links(artist, track, csv_track_link=csv_track_link)
    track_link = spotify["track_link"]
    album_link = spotify["album_link"]

    lyrics_link = ""
    if lyrics_entry:
        genius_url = str(lyrics_entry.get("genius_url", "")).strip()
        if genius_url.startswith("http") and "genius.com/" in genius_url and "-lyrics" in genius_url:
            lyrics_link = genius_url
    if not lyrics_link:
        lyrics_link = genius_direct_song_link(artist, track)

    return {
        "track_link": track_link,
        "album_link": album_link,
        "lyrics_link": lyrics_link,
        "track_source": "Spotify",
        "album_source": "Spotify",
        "lyrics_source": "Genius",
    }


def build_similarity_engine(df, lyrics_df=None):
    if df.empty:
        return None

    required = {"artist", "track_name", "genre"}
    if not required.issubset(df.columns):
        return None

    safe_df = df.copy().reset_index(drop=True)
    safe_df["artist"] = safe_df["artist"].fillna("").astype(str)
    safe_df["track_name"] = safe_df["track_name"].fillna("").astype(str)
    safe_df["genre"] = safe_df["genre"].fillna("").astype(str)
    if "spotify_link" not in safe_df.columns:
        safe_df["spotify_link"] = ""
    else:
        safe_df["spotify_link"] = safe_df["spotify_link"].fillna("").astype(str)
    safe_df["song_key"] = safe_df.apply(lambda r: song_key(r.get("artist", ""), r.get("track_name", "")), axis=1)

    lyrics_join = pd.DataFrame(columns=["song_key", "lyrics", "genius_url", "status"])
    if lyrics_df is not None and not lyrics_df.empty:
        working_lyrics = lyrics_df.copy()
        if "song_key" not in working_lyrics.columns and {"artist", "track_name"}.issubset(working_lyrics.columns):
            working_lyrics["song_key"] = working_lyrics.apply(
                lambda r: song_key(r.get("artist", ""), r.get("track_name", "")), axis=1
            )
        if "lyrics" not in working_lyrics.columns:
            working_lyrics["lyrics"] = ""
        if "genius_url" not in working_lyrics.columns:
            working_lyrics["genius_url"] = ""
        if "status" not in working_lyrics.columns:
            working_lyrics["status"] = ""
        lyrics_join = working_lyrics[["song_key", "lyrics", "genius_url", "status"]].copy()
        lyrics_join["lyrics"] = lyrics_join["lyrics"].fillna("").astype(str)
        lyrics_join["genius_url"] = lyrics_join["genius_url"].fillna("").astype(str)
        lyrics_join["status"] = lyrics_join["status"].fillna("").astype(str)
        lyrics_join = lyrics_join.drop_duplicates(subset=["song_key"], keep="last")

    safe_df = safe_df.merge(lyrics_join, on="song_key", how="left")
    safe_df["lyrics"] = safe_df["lyrics"].fillna("").astype(str)
    safe_df["genius_url"] = safe_df["genius_url"].fillna("").astype(str)
    safe_df["lyrics_status"] = safe_df["status"].fillna("").astype(str)
    if "status" in safe_df.columns:
        safe_df = safe_df.drop(columns=["status"])

    corpus = (
        safe_df["artist"]
        + " "
        + safe_df["track_name"]
        + " "
        + safe_df["genre"]
        + " "
        + safe_df["lyrics"]
    ).str.lower()

    vectorizer = TfidfVectorizer(max_features=4000, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(corpus)
    vectors = tfidf_matrix.astype(np.float32).toarray()

    backend = "sklearn"
    index = None
    nn = None
    query_vectors = vectors.copy()

    if vectors.size == 0:
        return None

    if faiss is not None:
        faiss.normalize_L2(query_vectors)
        index = faiss.IndexFlatIP(query_vectors.shape[1])
        index.add(query_vectors)
        backend = "faiss"
    else:
        nn = NearestNeighbors(metric="cosine", algorithm="brute")
        nn.fit(query_vectors)

    lyrics_lookup = build_lyrics_lookup(lyrics_df)

    metadata = []
    lyrics_available = 0
    for _, row in safe_df.iterrows():
        artist = row["artist"].strip()
        track = row["track_name"].strip()
        csv_link = row.get("spotify_link", "")
        has_lyrics = bool(str(row.get("lyrics", "")).strip())
        if has_lyrics:
            lyrics_available += 1
        key = song_key(artist, track)
        links = resolve_song_links(
            artist,
            track,
            csv_track_link=csv_link,
            lyrics_entry=lyrics_lookup.get(key),
        )
        metadata.append(
            {
                "artist": artist,
                "track": track,
                "genre": row["genre"].strip(),
                "spotify_link": links["track_link"],
                "album_link": links["album_link"],
                "lyrics_link": links["lyrics_link"],
                "track_source": links["track_source"],
                "album_source": links["album_source"],
                "lyrics_source": links["lyrics_source"],
                "has_lyrics": has_lyrics,
            }
        )

    return {
        "backend": backend,
        "vectorizer": vectorizer,
        "vectors": query_vectors,
        "index": index,
        "nn": nn,
        "metadata": metadata,
        "lyrics_coverage": lyrics_available,
        "total_songs": len(metadata),
    }


def get_similarity_engine(df):
    if df.empty:
        return None

    try:
        mtime = os.path.getmtime(FILE_PATH)
    except OSError:
        mtime = None
    try:
        lyrics_mtime = os.path.getmtime(LYRICS_DB_PATH)
    except OSError:
        lyrics_mtime = None

    cached_engine = _SIMILARITY_CACHE.get("engine")
    cached_source_mtime = _SIMILARITY_CACHE.get("source_mtime")
    cached_lyrics_mtime = _SIMILARITY_CACHE.get("lyrics_mtime")

    if cached_engine is not None and cached_source_mtime == mtime and cached_lyrics_mtime == lyrics_mtime:
        return cached_engine

    lyrics_df = load_lyrics_database(LYRICS_DB_PATH)
    engine = build_similarity_engine(df, lyrics_df=lyrics_df)
    _SIMILARITY_CACHE["engine"] = engine
    _SIMILARITY_CACHE["source_mtime"] = mtime
    _SIMILARITY_CACHE["lyrics_mtime"] = lyrics_mtime
    return engine


def find_similar_songs(engine, artist, track, top_k=5):
    if not engine:
        return [], {"backend": "none", "note": "Silnik podobienstwa nie jest gotowy."}

    metadata = engine["metadata"]
    if not metadata:
        return [], {"backend": engine["backend"], "note": "Brak metadanych."}

    query_key = f"{artist}::{track}".lower().strip()
    query_idx = None
    for idx, row in enumerate(metadata):
        row_key = f"{row['artist']}::{row['track']}".lower().strip()
        if row_key == query_key:
            query_idx = idx
            break

    if query_idx is None:
        query_idx = 0

    max_neighbors = min(len(metadata), top_k + 1)
    candidates = []

    if engine["backend"] == "faiss":
        scores, indices = engine["index"].search(engine["vectors"][query_idx: query_idx + 1], max_neighbors)
        for pos, cand_idx in enumerate(indices[0]):
            if int(cand_idx) == query_idx:
                continue
            score = float(scores[0][pos])
            row = metadata[int(cand_idx)].copy()
            row["score"] = score
            candidates.append(row)
    else:
        distances, indices = engine["nn"].kneighbors(engine["vectors"][query_idx: query_idx + 1], n_neighbors=max_neighbors)
        for pos, cand_idx in enumerate(indices[0]):
            if int(cand_idx) == query_idx:
                continue
            score = float(1.0 - distances[0][pos])
            row = metadata[int(cand_idx)].copy()
            row["score"] = score
            candidates.append(row)

    candidates = sorted(candidates, key=lambda x: x.get("score", 0.0), reverse=True)[:top_k]
    coverage_num = int(engine.get("lyrics_coverage", 0) or 0)
    coverage_den = int(engine.get("total_songs", 0) or 0)
    coverage_pct = (coverage_num / coverage_den * 100.0) if coverage_den > 0 else 0.0
    debug_payload = {
        "engine_label": "Szybki silnik FAISS" if engine["backend"] == "faiss" else "Tryb zgodnosci (sklearn)",
        "query_song": f"{metadata[query_idx]['artist']} - {metadata[query_idx]['track']}",
        "recommendation_count": top_k,
        "lyrics_coverage": f"{coverage_num}/{coverage_den}",
        "lyrics_coverage_pct": coverage_pct,
    }
    return candidates, debug_payload

def analyze_mood_with_ai(client, user_mood):
    """Asks OpenAI for valence and energy values from a natural-language mood."""
    prompt_template = PROMPTS.get("mood_analysis_user", "{user_mood}")
    prompt = str(prompt_template).replace("{user_mood}", user_mood)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        data = json.loads(response.choices[0].message.content or "{}")
        return data.get("valence", 0.5), data.get("energy", 0.5), data.get("diagnosis", "Nieznany nastroj")
    except Exception:
        return 0.5, 0.5, "Błąd AI"


def analyze_mood_locally(user_mood):
    """Local fallback if API key is missing or AI call fails."""
    tokens = re.findall(r"\w+", user_mood.lower())

    valence_map = {
        "szczęśliwa": 0.85, "szczęśliwy": 0.85, "wesoła": 0.8, "wesoły": 0.8,
        "euforia": 0.95, "radość": 0.85, "smutna": 0.2, "smutny": 0.2,
        "złamana": 0.15, "złamany": 0.15, "spokojna": 0.55, "spokojny": 0.55,
        "pewna": 0.75, "pewny": 0.75, "romantyczna": 0.7, "romantyczny": 0.7,
        "zła": 0.2, "zły": 0.2
    }

    energy_map = {
        "siłownia": 0.9, "trening": 0.9, "impreza": 0.95, "dance": 0.9,
        "chill": 0.35, "spokojna": 0.35, "spokojny": 0.35, "senna": 0.2,
        "senny": 0.2, "zmęczona": 0.25, "zmęczony": 0.25, "motywacja": 0.8,
        "wkurzona": 0.85, "wkurzony": 0.85
    }

    valence_scores = [valence_map[t] for t in tokens if t in valence_map]
    energy_scores = [energy_map[t] for t in tokens if t in energy_map]

    valence = sum(valence_scores) / len(valence_scores) if valence_scores else 0.5
    energy = sum(energy_scores) / len(energy_scores) if energy_scores else 0.5

    if valence >= 0.65 and energy >= 0.65:
        diagnosis = "wysoka energia"
    elif valence < 0.4 and energy < 0.45:
        diagnosis = "melancholijny chill"
    elif energy >= 0.75:
        diagnosis = "mocny boost"
    elif valence >= 0.65:
        diagnosis = "pozytywny vibe"
    else:
        diagnosis = "neutralny nastrój"

    return float(valence), float(energy), diagnosis

def filter_by_category(df, category_name):
    """Filters songs by selected genre category."""
    keywords = GENRE_CATEGORIES.get(category_name, [])

    if not keywords:
        return df

    pattern = "|".join(keywords)
    filtered_df = df[df["genre"].astype(str).str.contains(pattern, case=False, na=False)]

    return filtered_df

def find_best_songs(df, target_valence, target_energy, limit=5, lyrics_lookup=None):
    """Finds songs closest to requested mood metrics."""
    if df.empty:
        return []

    working_df = df.copy()
    working_df["distance"] = (abs(working_df["valence"] - target_valence) * 1.5 + abs(working_df["energy"] - target_energy))

    candidates = working_df.sort_values("distance").head(30)

    if not candidates.empty:
        sampled = candidates.sample(n=min(len(candidates), limit))
        songs = []
        for _, row in sampled.iterrows():
            artist = str(row.get("artist", "")).strip()
            track = str(row.get("track_name", "")).strip()
            genre_tag = str(row.get("genre", "")).strip()
            csv_link = str(row.get("spotify_link", "")).strip()
            lookup = lyrics_lookup or {}
            links = resolve_song_links(
                artist,
                track,
                csv_track_link=csv_link,
                lyrics_entry=lookup.get(song_key(artist, track)),
            )

            songs.append({
                "artist": artist,
                "track": track,
                "genre": genre_tag,
                "link": links["track_link"],
                "album_link": links["album_link"],
                "lyrics_link": links["lyrics_link"],
                "track_source": links["track_source"],
                "album_source": links["album_source"],
                "lyrics_source": links["lyrics_source"],
            })
        return songs
    return []


@app.route("/", methods=["GET", "POST"])
def index():
    full_df = load_song_data()
    lyrics_df = load_lyrics_database(LYRICS_DB_PATH)
    lyrics_lookup = build_lyrics_lookup(lyrics_df)

    selected_category = "Wszystkie / Dowolny"
    mood = ""
    num_songs = 5
    playlist = []
    diagnosis = None
    valence = None
    energy = None
    warning = None
    similar_tracks = []
    similarity_debug = None
    similarity_seed = None

    if full_df.empty:
        warning = f"Nie znaleziono pliku {FILENAME_CSV} albo plik jest pusty."

    if request.method == "POST":
        mood = (request.form.get("mood") or "").strip()
        selected_category = request.form.get("category", selected_category)
        try:
            num_songs = int(request.form.get("num_songs", num_songs))
        except ValueError:
            num_songs = 5

        num_songs = max(1, min(10, num_songs))

        if not mood:
            warning = "Napisz chociaz jedno slowo o tym, jak sie czujesz."
        elif full_df.empty:
            warning = f"Nie znaleziono pliku {FILENAME_CSV} albo plik jest pusty."
        else:
            filtered_df = filter_by_category(full_df, selected_category)
            if filtered_df.empty:
                warning = f"Nie znaleziono piosenek dla kategorii: {selected_category}."
            else:
                client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None
                if client:
                    valence, energy, diagnosis = analyze_mood_with_ai(client, mood)
                else:
                    valence, energy, diagnosis = analyze_mood_locally(mood)

                playlist = find_best_songs(
                    filtered_df,
                    valence,
                    energy,
                    limit=num_songs,
                    lyrics_lookup=lyrics_lookup,
                )
                if not playlist:
                    warning = "Brak piosenek pasujacych do nastroju. Sprobuj innego opisu."
                else:
                    similarity_engine = get_similarity_engine(full_df)
                    seed_song = playlist[0]
                    similar_tracks, similarity_debug = find_similar_songs(
                        similarity_engine,
                        seed_song["artist"],
                        seed_song["track"],
                        top_k=5,
                    )
                    similarity_seed = f"{seed_song['artist']} - {seed_song['track']}"

    return render_template(
        "index.html",
        categories=list(GENRE_CATEGORIES.keys()),
        selected_category=selected_category,
        mood=mood,
        num_songs=num_songs,
        playlist=playlist,
        diagnosis=diagnosis,
        valence=valence,
        energy=energy,
        warning=warning,
        similar_tracks=similar_tracks,
        similarity_debug=similarity_debug,
        similarity_seed=similarity_seed,
        ai_mode="AI online (OpenAI)" if OPENAI_KEY else "Lokalny fallback (bez API key)",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
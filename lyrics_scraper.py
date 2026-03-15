import re
import time
import urllib.parse
import json
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup

DEFAULT_TIMEOUT = 12
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}


def normalize_token(text: str) -> str:
    value = str(text or "").lower()
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r"[^a-z0-9 ]", "", value)
    return value


def _token_set(text: str) -> set[str]:
    value = normalize_token(text)
    if not value:
        return set()
    return {token for token in value.split(" ") if token}


def _overlap_ratio(target_tokens: set[str], found_tokens: set[str]) -> float:
    if not target_tokens or not found_tokens:
        return 0.0
    return len(target_tokens.intersection(found_tokens)) / len(target_tokens)


def song_key(artist: str, track: str) -> str:
    return f"{normalize_token(artist)}::{normalize_token(track)}"


def genius_search_url(artist: str, track: str) -> str:
    query = urllib.parse.quote(f"{artist} {track} lyrics")
    return f"https://genius.com/search?q={query}"


def fetch_page(url: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    response = requests.get(url, timeout=timeout, headers=REQUEST_HEADERS)
    response.raise_for_status()
    return response.text


def fetch_search_page(artist: str, track: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    return fetch_page(genius_search_url(artist, track), timeout=timeout)


def extract_result_links(search_html: str, max_links: int = 10) -> list[str]:
    """Extracts likely Genius song links from search result HTML."""
    soup = BeautifulSoup(search_html, "html.parser")
    links = []
    for anchor in soup.select("a[href]"):
        href = (anchor.get("href") or "").strip()
        if not href.startswith("http"):
            continue
        if "genius.com" not in href:
            continue
        if "-lyrics" not in href:
            continue
        if href not in links:
            links.append(href)
        if len(links) >= max_links:
            break
    return links


def pick_best_result_link(candidate_links: list[str], artist: str, track: str) -> str:
    if not candidate_links:
        return ""

    artist_norm = normalize_token(artist)
    track_norm = normalize_token(track)

    scored = []
    for link in candidate_links:
        link_norm = normalize_token(link.replace("-lyrics", " "))
        score = 0
        if artist_norm and artist_norm in link_norm:
            score += 2
        if track_norm and track_norm in link_norm:
            score += 2
        if "lyrics" in link.lower():
            score += 1
        scored.append((score, link))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def extract_song_meta_from_html(song_html: str) -> tuple[str, str]:
    """Returns (artist, title) extracted from Genius page metadata."""
    soup = BeautifulSoup(song_html, "html.parser")

    for node in soup.select("script[type='application/ld+json']"):
        raw = (node.string or "").strip()
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                nodes = parsed
            else:
                nodes = [parsed]

            for item in nodes:
                if not isinstance(item, dict):
                    continue
                title = str(item.get("name", "")).strip()
                by_artist = item.get("byArtist", {})
                artist = ""
                if isinstance(by_artist, dict):
                    artist = str(by_artist.get("name", "")).strip()
                if title or artist:
                    return artist, title
        except Exception:
            continue

    og_title = soup.select_one("meta[property='og:title']")
    if og_title and og_title.get("content"):
        content = str(og_title.get("content")).strip()
        content = content.replace("Lyrics", "").replace("| Genius Lyrics", "").strip()
        # often format: Artist - Track
        if " - " in content:
            split = content.split(" - ", 1)
            return split[0].strip(), split[1].strip()
        return "", content

    return "", ""


def metadata_match_score(target_artist: str, target_track: str, found_artist: str, found_track: str) -> float:
    target_artist_tokens = _token_set(target_artist)
    target_track_tokens = _token_set(target_track)
    found_artist_tokens = _token_set(found_artist)
    found_track_tokens = _token_set(found_track)

    score = 0.0
    if target_artist_tokens and found_artist_tokens:
        score += 2.0 * (len(target_artist_tokens.intersection(found_artist_tokens)) / len(target_artist_tokens))
    if target_track_tokens and found_track_tokens:
        score += 2.0 * (len(target_track_tokens.intersection(found_track_tokens)) / len(target_track_tokens))

    # bonus for exact normalized containment
    target_artist_norm = normalize_token(target_artist)
    target_track_norm = normalize_token(target_track)
    found_artist_norm = normalize_token(found_artist)
    found_track_norm = normalize_token(found_track)
    if target_artist_norm and target_artist_norm in found_artist_norm:
        score += 0.8
    if target_track_norm and target_track_norm in found_track_norm:
        score += 0.8

    return score


def url_match_score(target_artist: str, target_track: str, url: str) -> float:
    target_artist_tokens = _token_set(target_artist)
    target_track_tokens = _token_set(target_track)
    url_tokens = _token_set(url.replace("-lyrics", " "))
    if not url_tokens:
        return 0.0

    score = 0.0
    if target_artist_tokens:
        score += 1.6 * (len(target_artist_tokens.intersection(url_tokens)) / len(target_artist_tokens))
    if target_track_tokens:
        score += 1.6 * (len(target_track_tokens.intersection(url_tokens)) / len(target_track_tokens))
    return score


def extract_lyrics_from_song_page(song_html: str) -> str:
    """Extracts lyrics from Genius song page HTML."""
    soup = BeautifulSoup(song_html, "html.parser")

    container_nodes = soup.select("div[data-lyrics-container='true']")
    if container_nodes:
        chunks = [node.get_text("\n", strip=True) for node in container_nodes]
        lyrics = "\n".join([chunk for chunk in chunks if chunk])
        lyrics = re.sub(r"\n{2,}", "\n", lyrics).strip()
        return lyrics

    legacy_node = soup.select_one("div.lyrics")
    if legacy_node:
        return legacy_node.get_text("\n", strip=True)

    return ""


def scrape_lyrics_for_song(artist: str, track: str) -> dict:
    """Returns scraping payload for one song."""
    base = {
        "artist": str(artist or "").strip(),
        "track_name": str(track or "").strip(),
        "song_key": song_key(artist, track),
        "genius_url": "",
        "lyrics": "",
        "status": "not_found",
        "error": "",
        "updated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }

    try:
        search_html = fetch_search_page(artist, track)
        links = extract_result_links(search_html)
        if not links:
            raise RuntimeError("No Genius links in search response")

        best_link = ""
        best_lyrics = ""
        best_score = -1.0
        target_artist_tokens = _token_set(artist)
        target_track_tokens = _token_set(track)
        for link in links[:6]:
            try:
                song_html = fetch_page(link)
                found_artist, found_track = extract_song_meta_from_html(song_html)
                score_meta = metadata_match_score(artist, track, found_artist, found_track)
                score_url = url_match_score(artist, track, link)
                score = max(score_meta, score_url)

                found_tokens = _token_set(found_artist + " " + found_track + " " + link)
                artist_overlap = _overlap_ratio(target_artist_tokens, found_tokens)
                track_overlap = _overlap_ratio(target_track_tokens, found_tokens)

                # Require some overlap to avoid unrelated trending songs.
                if artist_overlap < 0.5 or track_overlap < 0.35:
                    continue

                if score < best_score:
                    continue
                lyrics = extract_lyrics_from_song_page(song_html)
                if not lyrics:
                    continue
                best_score = score
                best_link = link
                best_lyrics = lyrics
            except Exception:
                continue

        if not best_link or best_score < 1.0:
            raise RuntimeError("No confident Genius match")

        lyrics = best_lyrics
        if not lyrics:
            base["status"] = "empty_lyrics"
            base["genius_url"] = best_link
            return base

        base["genius_url"] = best_link
        base["lyrics"] = lyrics
        base["status"] = "ok"
        return base
    except Exception as genius_exc:
        # Fallback provider for real lyrics when Genius blocks scraping requests.
        try:
            ovh_url = f"https://api.lyrics.ovh/v1/{urllib.parse.quote(str(artist or ''))}/{urllib.parse.quote(str(track or ''))}"
            response = requests.get(ovh_url, timeout=DEFAULT_TIMEOUT, headers=REQUEST_HEADERS)
            response.raise_for_status()
            data = response.json()
            lyrics = str(data.get("lyrics", "")).strip()
            if lyrics:
                base["lyrics"] = lyrics
                base["status"] = "ok_ovh_fallback"
                base["error"] = f"genius_unavailable: {genius_exc}"
                base["genius_url"] = genius_search_url(artist, track)
                return base
        except Exception as ovh_exc:
            base["status"] = "error"
            base["error"] = f"genius: {genius_exc}; ovh: {ovh_exc}"
            return base

        base["status"] = "no_confident_match"
        base["error"] = str(genius_exc)
        return base


def load_lyrics_database(db_path: str) -> pd.DataFrame:
    if not db_path or not isinstance(db_path, str):
        return pd.DataFrame()
    if not pd.io.common.file_exists(db_path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(db_path)
        if "song_key" not in df.columns and {"artist", "track_name"}.issubset(df.columns):
            df["song_key"] = df.apply(lambda r: song_key(r.get("artist", ""), r.get("track_name", "")), axis=1)
        return df
    except Exception:
        return pd.DataFrame()


def save_lyrics_database(df: pd.DataFrame, db_path: str) -> None:
    out = df.copy()
    out.to_csv(db_path, index=False, encoding="utf-8")


def build_or_update_lyrics_database(
    songs_df: pd.DataFrame,
    db_path: str,
    limit: int | None = None,
    force_rescrape: bool = False,
    sleep_seconds: float = 0.8,
) -> pd.DataFrame:
    """Builds/updates local lyrics database for songs in songs_df."""
    if songs_df.empty or not {"artist", "track_name"}.issubset(songs_df.columns):
        return load_lyrics_database(db_path)

    existing_df = load_lyrics_database(db_path)
    existing_records = {}
    if not existing_df.empty and "song_key" in existing_df.columns:
        for _, row in existing_df.iterrows():
            existing_records[str(row.get("song_key", ""))] = row.to_dict()

    tasks = []
    seen = set()
    for _, row in songs_df.iterrows():
        artist = str(row.get("artist", "")).strip()
        track = str(row.get("track_name", "")).strip()
        if not artist or not track:
            continue
        key = song_key(artist, track)
        if key in seen:
            continue
        seen.add(key)
        if not force_rescrape and key in existing_records:
            current = existing_records[key]
            current_status = str(current.get("status", "")).lower()
            if current_status in {"ok", "ok_ovh_fallback"} and str(current.get("lyrics", "")).strip():
                continue
        tasks.append((artist, track))
        if limit is not None and limit > 0 and len(tasks) >= limit:
            break

    for idx, (artist, track) in enumerate(tasks, start=1):
        payload = scrape_lyrics_for_song(artist, track)
        existing_records[payload["song_key"]] = payload
        if idx < len(tasks):
            time.sleep(max(0.0, sleep_seconds))

    if existing_records:
        final_df = pd.DataFrame(existing_records.values())
        final_df = final_df.sort_values(["artist", "track_name"], na_position="last")
    else:
        final_df = pd.DataFrame(
            columns=["artist", "track_name", "song_key", "genius_url", "lyrics", "status", "error", "updated_at"]
        )

    save_lyrics_database(final_df, db_path)
    return final_df


if __name__ == "__main__":
    html = fetch_search_page("The Weeknd", "Starboy")
    links = extract_result_links(html, max_links=5)
    print("candidate links:", links)

import argparse
import os

import pandas as pd

from lyrics_scraper import build_or_update_lyrics_database

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SONGS_CSV = os.path.join(BASE_DIR, "baza_piosenek.csv")
LYRICS_DB = os.path.join(BASE_DIR, "lyrics_database.csv")


def parse_args():
    parser = argparse.ArgumentParser(description="Build/update lyrics database from Genius.")
    parser.add_argument("--limit", type=int, default=0, help="Max number of songs to scrape this run (0 = all missing).")
    parser.add_argument("--force", action="store_true", help="Force rescrape even when lyrics already exist.")
    parser.add_argument("--sleep", type=float, default=0.8, help="Delay between requests in seconds.")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(SONGS_CSV):
        raise FileNotFoundError(f"Missing source songs file: {SONGS_CSV}")

    songs_df = pd.read_csv(SONGS_CSV, on_bad_lines="skip")

    limit = args.limit if args.limit and args.limit > 0 else None
    out_df = build_or_update_lyrics_database(
        songs_df=songs_df,
        db_path=LYRICS_DB,
        limit=limit,
        force_rescrape=args.force,
        sleep_seconds=max(0.0, args.sleep),
    )

    total = len(out_df)
    ok_genius = int((out_df.get("status", "") == "ok").sum()) if total > 0 else 0
    ok_fallback = int((out_df.get("status", "") == "ok_ovh_fallback").sum()) if total > 0 else 0
    print(
        f"lyrics_database.csv updated: total={total}, ok_genius={ok_genius}, "
        f"ok_fallback={ok_fallback}, path={LYRICS_DB}"
    )


if __name__ == "__main__":
    main()

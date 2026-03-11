import streamlit as st
import pandas as pd
from openai import OpenAI
import json
import urllib.parse
import os

# ==========================================
# ⚙️ 1. KONFIGURACJA I KLUCZE
# ==========================================
# Sprawdzamy, czy działamy w chmurze, czy na komputerze
if "API_KEY" in st.secrets:
    OPENAI_KEY = st.secrets["API_KEY"]
else:
    try:
        from config import API_KEY
        OPENAI_KEY = API_KEY
    except ImportError:
        st.error("❌ Brak klucza API! Jeśli jesteś lokalnie: stwórz plik config.py. Jeśli w chmurze: ustaw Secrets.")
        st.stop()

# Ścieżka do pliku z bazą
FILENAME_CSV = "baza_piosenek.csv"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, FILENAME_CSV)

# ==========================================
# 🎵 2. DEFINICJA KATEGORII (Grupowanie)
# ==========================================
# Tutaj ustalamy, jakie słowa w bazie pasują do jakiej kategorii
GENRE_CATEGORIES = {
    "Wszystkie / Dowolny": [], 
    "Rap / Hip-Hop / Drill": ["rap", "hip hop", "hip-hop", "drill", "trap", "baddie", "gangsta", "old school"],
    "Pop / K-Pop": ["pop", "dance", "k-pop", "kpop", "korean", "mainstream"],
    "Rock / Metal / Alternatywa": ["rock", "metal", "punk", "grunge", "indie", "alternative"],
    "R&B / Soul": ["r&b", "rnb", "soul", "blues", "jazz", "chill"],
    "Elektroniczna / Club": ["house", "techno", "edm", "club", "electronic"]
}

# ==========================================
# 🧠 3. FUNKCJE (Mózg programu)
# ==========================================

def analyze_mood_with_ai(client, user_mood):
    """Pyta AI o parametry emocji na podstawie opisu"""
    prompt = f"""
    Jesteś profesjonalnym DJ-em. 
    Opis nastroju użytkownika: "{user_mood}"
    
    Twoim zadaniem jest określić dwa parametry muzyczne (0.0 do 1.0):
    1. Valence (Radość/Pozytywność): 0.0 to smutek/mrok, 1.0 to euforia/szczęście.
    2. Energy (Energia): 0.0 to senność/spokój, 1.0 to chaos/szybkość.
    
    Zwróć TYLKO format JSON: {{"valence": <float>, "energy": <float>, "diagnosis": "<krótki opis w 3 słowach>"}}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        data = json.loads(response.choices[0].message.content)
        return data.get('valence', 0.5), data.get('energy', 0.5), data.get('diagnosis', 'Nieznany nastrój')
    except Exception as e:
        return 0.5, 0.5, "Błąd AI"

def filter_by_category(df, category_name):
    """Filtruje tabelę, zostawiając tylko wybrane gatunki"""
    keywords = GENRE_CATEGORIES.get(category_name, [])
    
    # Jeśli wybrano "Wszystkie" (pusta lista keywords), zwracamy całą bazę
    if not keywords:
        return df
    
    # Szukamy, czy w kolumnie 'genre' występuje któreś ze słów kluczowych
    # (np. czy 'baddie rap' zawiera słowo 'rap')
    pattern = '|'.join(keywords) 
    filtered_df = df[df['genre'].astype(str).str.contains(pattern, case=False, na=False)]
    
    return filtered_df

def find_best_songs(df, target_valence, target_energy, limit=5):
    """Szuka piosenek najbliższych matematycznie do nastroju"""
    if df.empty:
        return pd.DataFrame()

    working_df = df.copy()
    # Obliczamy "odległość" emocjonalną
    working_df['distance'] = (abs(working_df['valence'] - target_valence) * 1.5 + abs(working_df['energy'] - target_energy))
    
    # Bierzemy 30 najlepszych kandydatów i losujemy z nich wybraną liczbę (żeby nie było nudno)
    candidates = working_df.sort_values('distance').head(30)
    
    if not candidates.empty:
        return candidates.sample(n=min(len(candidates), limit))
    else:
        return pd.DataFrame()

# ==========================================
# 🎨 4. WYGLĄD APLIKACJI (UI)
# ==========================================
st.set_page_config(page_title="MOAI 2026", page_icon="🎧", layout="centered")

# Tytuł i opis
st.title("🎧 MOAI 2026 - Twój AI DJ")
st.markdown("Wybierz kategorię, opisz vibe, a AI dobierze idealne kawałki.")

# Ładowanie bazy danych na start
try:
    full_df = pd.read_csv(FILE_PATH, on_bad_lines='skip')
except:
    st.error(f"⚠️ Błąd: Nie znaleziono pliku {FILENAME_CSV}. Wrzuć go na GitHub!")
    full_df = pd.DataFrame()

# Kontener z formularzem (dwa kafelki obok siebie)
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        mood = st.text_input("Jak się czujesz?", placeholder="np. pewna siebie, smutna, na siłownię")
    with col2:
        # Lista rozwijana z naszymi kategoriami
        category = st.selectbox("Wybierz gatunek:", list(GENRE_CATEGORIES.keys()))

    # Suwak
    num_songs = st.slider("Ile piosenek chcesz?", min_value=1, max_value=10, value=5)
    
    # Przycisk
    generate_btn = st.button("🎵 Generuj Playlistę", type="primary")

# ==========================================
# 🚀 5. LOGIKA PO KLIKNIĘCIU
# ==========================================
if generate_btn and mood:
    client = OpenAI(api_key=OPENAI_KEY)
    
    with st.spinner('🎧 AI przeszukuje bazę i analizuje Twój vibe...'):
        
        # KROK 1: Filtrowanie po kategorii (np. tylko Rap)
        filtered_df = filter_by_category(full_df, category)
        
        if filtered_df.empty:
            st.warning(f"⚠️ Nie znalazłam żadnych piosenek dla kategorii: {category}. Sprawdź czy w pliku CSV są dobre nazwy gatunków.")
        else:
            # KROK 2: Analiza AI (zamiana tekstu na liczby)
            valence, energy, diag = analyze_mood_with_ai(client, mood)
            
            # KROK 3: Szukanie piosenek
            playlist = find_best_songs(filtered_df, valence, energy, limit=num_songs)

            # --- Wyświetlanie wyników ---
            st.markdown("---")
            st.success(f"Twój Vibe: {diag.upper()}")
            
            # Kafelki z parametrami (dla ciekawskich)
            c1, c2, c3 = st.columns(3)
            c1.metric("Szukana Radość", f"{valence:.2f}")
            c2.metric("Szukana Energia", f"{energy:.2f}")
            c3.metric("Liczba wyników", len(playlist))
            
            st.subheader(f"🎹 Playlista ({category}):")
            
            if not playlist.empty:
                for index, row in playlist.iterrows():
                    # Pobieranie danych z wiersza
                    artist = row['artist']
                    track = row['track_name']
                    genre_tag = row['genre']
                    
                    # Tworzenie linku do Spotify (Search)
                    query = urllib.parse.quote(f"{artist} {track}")
                    link = f"https://open.spotify.com/search/{query}"
                    
                    # Wygląd pojedynczej piosenki
                    with st.container():
                        col_text, col_btn = st.columns([3, 1])
                        with col_text:
                            st.markdown(f"**{artist} - {track}**")
                            st.caption(f"🏷️ {genre_tag}")
                        with col_btn:
                            st.link_button("Odtwórz ▶️", link)
                        st.divider()
            else:
                st.warning("Znalazłam gatunek, ale żadna piosenka nie pasuje do tego nastroju. Spróbuj zmienić opis!")

elif generate_btn and not mood:
    st.warning("⚠️ Napisz chociaż jedno słowo o tym, jak się czujesz!")
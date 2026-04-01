import json
import os
import pandas as pd
import numpy as np
from typing import List

class CRAG(object):
    def __init__(self):
        print("\n" + "="*50)
        print("[DEBUG] INITIALIZING OFFLINE DATABASE ACCESS")
        print("="*50)
        self.data_dir = "models/processed_data"
        
        def load_csv(name):
            path = os.path.join(self.data_dir, name)
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path)
                    print(f"[DEBUG] SUCCESS: Loaded {name} ({len(df)} rows)")
                    return df
                except Exception as e:
                    print(f"[DEBUG] ERROR reading CSV {name}: {e}")
            print(f"[DEBUG] FAILED: {name} not found at {path}")
            return pd.DataFrame()

        def load_npy(name):
            path = os.path.join(self.data_dir, name)
            if os.path.exists(path):
                try:
                    data = np.load(path, allow_pickle=True)
                    # Perbaikan: Cek dimensi array untuk menghindari scalar error
                    if data.ndim == 0:
                        data = data.item()
                    print(f"[DEBUG] SUCCESS: Loaded {name} (Type: {type(data)})")
                    return data
                except Exception as e:
                    print(f"[DEBUG] ERROR loading NPY {name}: {e}")
                    return {}
            print(f"[DEBUG] FAILED: {name} not found at {path}")
            return {}

        self.oscar_df = load_csv("the_oscar_award.csv")
        self.grammy_df = load_csv("the_grammy_awards.csv")
        self.finance_data = load_npy("finance_data.npy")
        self.imdb_movies = load_npy("imdb_movie_dataset.npy")
        self.grammy_map = load_npy("grammy.npy")
        print("="*50 + "\n")

    def _empty(self):
        return {"result": []}

    def movie_get_movie_info(self, movie_name: str):
        print(f"[DEBUG] API Query: Searching movie info for '{movie_name}'")
        if not self.imdb_movies: return self._empty()
        query = str(movie_name).lower().strip()
        results = []
        if isinstance(self.imdb_movies, dict):
            results = [v for k, v in self.imdb_movies.items() if query in str(k).lower()]
        print(f"[DEBUG] API Response: Found {len(results)} matches.")
        return {"result": results} if results else self._empty()

    def movie_get_year_info(self, year: str):
        print(f"[DEBUG] API Query: Searching Oscar records for year {year}")
        if self.oscar_df.empty: return self._empty()
        try:
            df = self.oscar_df[self.oscar_df['year_ceremony'] == int(year)].copy()
            df['winner'] = df['winner'].astype(bool)
            awards = df[['category', 'name', 'film', 'winner']].to_dict('records')
            return {"result": {"oscar_awards": awards, "movie_list": []}}
        except: return self._empty()

    def finance_get_price_history(self, ticker: str): return self._empty()
    def finance_get_detailed_price_history(self, ticker: str): return self._empty()
    def finance_get_pe_ratio(self, ticker: str): return self._empty()
    def finance_get_market_capitalization(self, ticker: str): return self._empty()
    def finance_get_dividends_history(self, ticker: str): return self._empty()
    def finance_get_eps(self, ticker: str): return self._empty()
    def finance_get_company_name(self, query: str): return self._empty()
    def finance_get_ticker_by_name(self, query: str): return self._empty()
    def music_search_artist_entity_by_name(self, name: str): return self._empty()
    def music_search_song_entity_by_name(self, name: str): return self._empty()
    def music_get_artist_all_works(self, name: str): return self._empty()
    def sports_soccer_get_games_on_date(self, d, t): return self._empty()
    def sports_nba_get_games_on_date(self, d, t): return self._empty()
    def open_search_entity_by_name(self, q): return self._empty()
    def open_get_entity(self, e): return self._empty()
# AI-Powered Movie Recommendation System (Streamlit)

Content-based recommendation app that searches TMDB for a movie, builds a corpus from TMDB endpoints, and recommends similar titles using genres + overview TF-IDF cosine similarity. Displays posters, titles and genres.

## Features
- Search TMDB for a movie.
- Select the exact match.
- App fetches movie metadata + builds a corpus of movies from TMDB.
- Recommends similar movies (poster, title, genres).

## Files
- `app.py` — main Streamlit app.
- `requirements.txt` — Python packages.

## Setup (local)
1. Clone the repo.
2. Create & activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS / Linux
   venv\Scripts\activate    # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Obtain a TMDB API key at https://www.themoviedb.org/.
5. Run the app:
   ```bash
   export TMDB_API_KEY=your_key_here   # macOS / Linux
   set TMDB_API_KEY=your_key_here      # Windows (cmd)
   streamlit run app.py
   ```

## Deployment
- Push the repo to GitHub.
- Deploy on Streamlit Cloud by connecting your GitHub repo and setting `TMDB_API_KEY` as a secret, or paste the key in the UI for quick testing.

## Notes
- This demo uses TMDB (requires API key). For an offline reproducible variant, use MovieLens dataset.
# AI-RESUME-ASSISTANCE-
# 📄 ATS Resume Analyzer

Upload a resume (PDF, DOCX or TXT) and get an **ATS score**, strengths, weaknesses,
missing keywords, prioritised improvements and rewritten bullet examples.
Optionally paste a job description to see how well the resume matches it.

Built with [Streamlit](https://streamlit.io) and Google's **Gemini Flash** model.

## How the score works

The final score (0-100) blends two parts:

| Part | Weight | What it measures |
|------|--------|------------------|
| Rule-based checks | 40% | Contact info, standard sections, length, bullets, metrics, action verbs, text readability |
| Gemini AI review | 60% | Keyword/skills coverage, impact, clarity, and job-description match |

If the AI call fails (bad key, quota, network), the app still shows the rule-based score.

## Run locally

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Get a free Gemini API key at <https://aistudio.google.com/apikey>, then either paste it
into the sidebar or set it up as a secret (below).

### Configuring the API key

Pick one:

1. **Sidebar** - paste the key into the "Gemini API key" box (nothing is saved).
2. **Environment variable** - `export GEMINI_API_KEY="your-key"`
3. **Streamlit secrets** - create `.streamlit/secrets.toml`:
   ```toml
   GEMINI_API_KEY = "your-key"
   ```

> ⚠️ Never commit `secrets.toml` or your key to GitHub.

### Changing the model

The default is `gemini-flash-latest` (always the newest Flash model). You can change it in
the sidebar or with the `GEMINI_MODEL` environment variable, e.g. `gemini-2.5-flash`.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (public, or private with access granted).
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. Click **Create app** → **Deploy a public app from GitHub**.
4. Choose your repository, branch `main`, and main file path `app.py`.
5. Open **Advanced settings → Secrets** and paste:
   ```toml
   GEMINI_API_KEY = "your-key"
   ```
6. Click **Deploy**.

> If your app is public and you store your key in Secrets, anyone using the app spends your
> Gemini quota. To avoid this, leave the secret out so each visitor enters their own key.

## Privacy

Resumes are processed in memory and sent to the Gemini API for analysis. The app does not
store uploaded files.

## Limitations

- Scanned/image-only PDFs have no extractable text; export a text-based PDF instead.
- The ATS score is an estimate. Real ATS systems differ, so treat it as guidance.
- AI suggestions can be imperfect. Never add experience or metrics that aren't true.

## Project structure

```
app.py             # Streamlit app (UI + scoring + Gemini calls)
requirements.txt   # Python dependencies
README.md          # This file
```

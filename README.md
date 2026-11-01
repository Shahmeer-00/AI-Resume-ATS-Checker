# 📄 AI Resume ATS Checker

Upload a resume (PDF, DOCX or TXT) and get an **ATS score**, strengths and weaknesses, **prioritised improvements with example rewrites**, and a **keyword gap analysis** against an optional job description. Built with **Python**, **Streamlit** and **Google Gemini Flash**.

## Features

- Reads PDF, DOCX and TXT resumes (shows the exact text an ATS can extract).
- **Two-part score** so results are explainable:
  - **Rule-based checks (40%)**: contact details, section headings, length, bullet points, measurable results, action verbs, dates, readable text, and job-description keyword match.
  - **Gemini AI review (60%)**: relevance, structure, content quality, impact and language, plus concrete fixes.
- If the AI is unavailable (no key, quota, network), the app still shows the rule-based score.
- Optional job title and job description for tailored feedback.
- Downloadable Markdown report.
- Friendly error messages (bad key, rate limit, scanned PDF, password-protected PDF, corrupt file).

> No tool can reproduce any specific company's ATS exactly. Treat the score as guidance, not a guarantee.

## Project Structure

```
ats-resume-checker/
├── app.py            # Whole application
├── requirements.txt  # Dependencies
└── README.md         # This file
```

## Prerequisites

- Python 3.9+
- A free **Gemini API key** from [Google AI Studio](https://aistudio.google.com/app/apikey)
- Git and a GitHub account (for deployment)

## Run Locally

```bash
cd ats-resume-checker

# 1. Virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 2. Install
pip install -r requirements.txt

# 3. (Optional) set the key so you don't paste it each time
set GEMINI_API_KEY=your_key_here          # Windows (cmd)
# export GEMINI_API_KEY=your_key_here     # macOS / Linux

# 4. Run
streamlit run app.py
```

You can also paste the key into the app's sidebar. The model name is also editable in the sidebar (default `gemini-2.5-flash`). If a model name is not found, the app automatically tries `gemini-3.6-flash` and `gemini-2.5-flash`. Google renames models over time, so confirm the current Flash model name in AI Studio and set it in the sidebar, or via the `GEMINI_MODEL` secret / environment variable.

## Push to GitHub

1. Create a new repository on GitHub (for example `ats-resume-checker`). Public is required for the free Streamlit Community Cloud plan unless you connect a private repo.
2. In your project folder:

```bash
git init
git add app.py requirements.txt README.md
git commit -m "AI Resume ATS Checker"
git branch -M main
git remote add origin https://github.com/<your-username>/ats-resume-checker.git
git push -u origin main
```

**Never commit your API key.** Do not put it inside `app.py`. If you use a local secrets file (`.streamlit/secrets.toml`), keep it out of Git with a `.gitignore` containing:

```
.streamlit/secrets.toml
venv/
__pycache__/
```

## Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **Create app** > **Deploy a public app from GitHub**.
3. Choose your repository, branch `main`, and main file path `app.py`.
4. Open **Advanced settings** > **Secrets** and paste:

```toml
GEMINI_API_KEY = "your_key_here"
# GEMINI_MODEL = "gemini-2.5-flash"   # optional
```

5. Click **Deploy**. The first build installs `requirements.txt` (a few minutes).
6. Open the app URL. You can later edit secrets in **App settings > Secrets**. Pushing to GitHub redeploys automatically.

If you do not add a secret, each visitor can paste their own key in the sidebar.

## How the Score Works

| Part | Weight | What it checks |
|---|---|---|
| Rule-based | 40% | Email, phone, links, Summary/Experience/Education/Skills headings, length (400-900 words ideal), bullets, numbers/metrics, action verbs, dates, clean text, job-description keywords |
| Gemini AI | 60% | Overall ATS readiness, relevance to the role, content quality, impact, clarity, and specific fixes |

Labels: 80+ Excellent, 65-79 Good, 50-64 Needs work, below 50 Poor.

## Privacy

Resume text is sent to Google's Gemini API to produce the AI review. The app does not store resumes or keys; results exist only in your browser session. Do not upload documents you are not comfortable sending to Google.

## Troubleshooting

| Problem | Fix |
|---|---|
| "Very little text could be read" | The PDF is a scanned image or design-heavy. Export a text-based PDF or DOCX (this also helps real ATS parsing). |
| "API key was rejected" | Re-copy the key from AI Studio; make sure no spaces; confirm the Gemini API is enabled. |
| "Rate limit or free quota reached" | Wait a minute or use another key; the free tier has per-minute and daily limits. |
| "Model name was not found" | Put a current Flash model name from AI Studio in the sidebar. |
| `.doc` file rejected | Save it as `.docx` or PDF. |
| Deploy fails on Streamlit Cloud | Make sure `requirements.txt` is in the repo root and the main file is `app.py`. |
| `streamlit: command not found` | Activate the virtual environment and reinstall requirements. |

## Future Improvements

- Support scanned resumes with OCR.
- Side-by-side comparison of two resume versions.
- Cover letter and LinkedIn summary generator.
- Export the report as PDF.

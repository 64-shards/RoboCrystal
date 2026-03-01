# RoboCrystal Deployment Guide (Streamlit-first)

You asked for a simpler workflow than managing a cloud VM. This guide makes **local run** and
**Streamlit Community Cloud** the primary deployment paths.

---

## Option A (recommended): Run locally in 5 minutes

### 1) Open a terminal in the project directory

```bash
cd /workspace/RoboCrystal
```

### 2) Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4) Set runtime secrets (optional but recommended)

Create `.env` from template:

```bash
cp .env.example .env
```

Then fill values:

```dotenv
MONGO_URI=<your_mongo_uri>
MONGO_DB=bipedal_parity
MONGO_COLLECTION=historical_costs
GEMINI_API_KEY=<your_gemini_key>
```

If you skip secrets, the app still runs and falls back to local CSV/pipeline behavior.

### 5) Start the app

```bash
streamlit run app.py
```

Open this in your browser:

- `http://localhost:8501`

### 6) Stop the app

Press `Ctrl + C` in the terminal running Streamlit.

---

## Option B: Public URL with Streamlit Community Cloud (no VM, no TLS setup)

### 1) Push this repo to GitHub

If this repo isn't on GitHub yet, create one and push your branch.

### 2) Open Streamlit Community Cloud

- Visit: <https://share.streamlit.io>
- Sign in with GitHub

### 3) Deploy app

- Click **New app**
- Select your repository and branch
- Set main file path to: `app.py`
- Click **Deploy**

### 4) Add secrets in Streamlit Cloud

In app settings, add secrets equivalent to `.env.example`:

- `MONGO_URI`
- `MONGO_DB`
- `MONGO_COLLECTION`
- `GEMINI_API_KEY`

Then reboot/redeploy the app.

### 5) Validate

- App loads at public Streamlit URL
- Charts render
- Data source + Gemini features work when secrets are set

---

## Security notes

- `.env` must stay uncommitted (already ignored by `.gitignore`).
- Prefer read-only MongoDB credentials for dashboard use.
- Restrict MongoDB Atlas network access where possible.

---

## Update workflow

### Local

```bash
git pull
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

### Streamlit Community Cloud

- Push new commits to configured branch
- Cloud app auto-rebuilds (or click **Reboot app**)

---

## Rollback workflow

```bash
git log --oneline
git checkout <known_good_commit>
source .venv/bin/activate
streamlit run app.py
```

If using Streamlit Cloud, push that known-good commit (or reset branch) and redeploy.

---

## Optional: keep Docker-based self-hosting later

The repository still includes `Dockerfile`, `docker-compose.yml`, and `Caddyfile` if you decide to
return to self-hosted deployment in the future.

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
# RoboCrystal Deployment Guide (Vultr + Docker + Caddy)

This guide deploys `app.py` to a public HTTPS URL with runtime secrets.

## 1) Provision infrastructure on Vultr

1. Create a **Cloud Compute** instance:
   - OS: **Ubuntu 22.04 LTS**
   - Size: **1 vCPU / 2GB RAM** or larger
2. Reserve/record the instance public IPv4 address.
3. Create/apply a Vultr Firewall Group to this instance allowing inbound TCP only:
   - `22` (SSH)
   - `80` (HTTP)
   - `443` (HTTPS)

## 2) Configure DNS

Create an A record for your app host (example uses `app.example.com`) and point it to your Vultr public IP.

```bash
dig app.<yourdomain.com> +short
```

Wait until the command returns your server IPv4 before proceeding.

## 3) Prepare server runtime

SSH into the server:

```bash
ssh root@<server-ip>
```

Install Docker Engine + Compose plugin:

```bash
apt-get update
apt-get install -y ca-certificates curl gnupg
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo $VERSION_CODENAME) stable" \
  > /etc/apt/sources.list.d/docker.list

apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
systemctl enable --now docker
```

Create deploy directory:

```bash
mkdir -p /opt/robocrystal
```

## 4) Ship the repository

Option A (recommended):

```bash
cd /opt/robocrystal
git clone <your-repo-url> .
```

Option B: upload files via `scp`/CI artifact.

## 5) Configure runtime secrets (do not commit)

Create `/opt/robocrystal/.env`:

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
The app reads these values at runtime (`os.getenv` in `app.py`).

## 6) Start services

From `/opt/robocrystal`:

```bash
docker compose up -d --build
docker compose ps
docker compose logs --tail=200 app caddy
```

## 7) Validate public HTTPS and app health

Open:

- `https://app.<yourdomain.com>`

Checks:
- Browser shows valid TLS certificate (issued by Caddy/Let's Encrypt).
- Streamlit dashboard loads.
- Data-source operations succeed with configured secrets.

Confirm Streamlit is not directly exposed on public network:

```bash
ss -tulpn | rg 8501
```

Expected: no host-level `0.0.0.0:8501` listener (only Docker internal networking).

## 8) Operational hardening

### Container restart policy
Already set to `restart: unless-stopped` for both services in `docker-compose.yml`.

### Docker log rotation
Create `/etc/docker/daemon.json`:

```json
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
```

Then restart Docker:

```bash
systemctl restart docker
```

### MongoDB hardening
- Restrict MongoDB Atlas Network Access to your Vultr IPv4.
- Use least-privilege MongoDB credentials (read-only for dashboard path).

### Uptime monitoring
Use Uptime Kuma, Better Stack, or similar to monitor:

- `https://app.<yourdomain.com>`

## 9) Release/update workflow

From `/opt/robocrystal`:

```bash
git pull
docker compose up -d --build
docker compose logs -f app
```

## 10) Rollback procedure

```bash
cd /opt/robocrystal
git log --oneline
# choose a known-good commit:
git checkout <commit_sha>
docker compose up -d --build
```

To return to latest mainline later:

```bash
git checkout <branch>
git pull
docker compose up -d --build
```

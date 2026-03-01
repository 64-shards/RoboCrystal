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

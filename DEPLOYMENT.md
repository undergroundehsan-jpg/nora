# NORA deployment

Backend on Railway (Docker), voice frontend and analytics dashboard on Netlify.

| Piece | URL | Repository |
| --- | --- | --- |
| Backend | <https://nora-production-ae4f.up.railway.app> | `FINOVA_AWAAZ_AI` (root) |
| Voice / call testing | <https://starlit-boba-624371.netlify.app> | `FINOVA_AWAAZ_AI` → base `frontend` |
| Analytics dashboard | <https://amazing-mooncake-6b9ea6.netlify.app> | **its own repo** in `dashboard/` |

`dashboard/` is a separate git repository, so its changes must be committed and
pushed from inside that folder — the root repo does not contain it.

---

## 0. Commit what the server needs

The audio cache and the customer record are currently untracked, and the server
cannot work without them: a missing `audio_cache/manifest.json` silently turns
every pre-recorded line into a live text-to-speech call.

```bash
git add Dockerfile .dockerignore railway.json requirements-deploy.txt DEPLOYMENT.md
git add data/leads.json audio_cache/
git add frontend/ dashboard/netlify.toml app/
git status                      # confirm .env is NOT listed
git commit -m "Add Railway/Netlify deployment setup and phone call button"
git push origin main
```

`.env` is in `.gitignore` and must stay out of the repository. Secrets go into
Railway's Variables tab instead.

---

## 1. Backend on Railway

### 1.1 Create the service

1. Go to <https://railway.com> → **New Project** → **Deploy from GitHub repo**.
2. Pick this repository and the `main` branch.
3. Railway reads `railway.json`, sees `"builder": "DOCKERFILE"` and builds the
   `Dockerfile`. The first build takes roughly 5–8 minutes because the CPU build
   of PyTorch is about 200 MB.

### 1.2 Set the environment variables

**Variables** tab → **Raw Editor** → paste this, filling in the real secrets
(they are in your local `.env`):

```env
GROQ_API_KEY=gsk_...
STT_DEEPGRAM=...
RIME_API_KEY=...

RIME_MODEL=arcana
RIME_VOICE=cupola
RIME_LANG=eng
RIME_GENRE=conversational
RIME_SPEED_ALPHA=1.0
RIME_TIME_SCALE_FACTOR=1.0
RIME_REPETITION_PENALTY=1.1
RIME_AUDIO_FORMAT=wav
RIME_SAMPLING_RATE=16000

TWILIO_ENABLED=true
TWILIO_ACCOUNT_SID=AC...
TWILIO_AUTH_TOKEN=...
TWILIO_NUMBER=+19069848267
TWILIO_OUTBOUND_DEFAULT_TO=+923325026869
TELEPHONY_DEFAULT_PROVIDER=twilio
RINGCX_ENABLED=false

ALLOW_SAMPLE_LEAD=true
```

Leave `PORT` alone — Railway sets it and the container already uses it.

### 1.3 Get the domain, then point the app at itself

1. **Settings → Networking → Generate Domain**. You get something like
   `nora-backend-production.up.railway.app`.
2. Add one more variable:
   `PUBLIC_BASE_URL=https://nora-backend-production.up.railway.app`
3. Redeploy. This value is what Twilio is told to fetch TwiML from and what the
   media stream connects back to, so the first deploy cannot know it.

### 1.4 Two settings that matter

- **Replicas = 1** (Settings → Deploy). Call sessions live in process memory, so
  the REST call that starts a call and the Twilio media socket must reach the
  same instance. More than one replica breaks calls at random.
- **Volume mounted at `/app/logs`** (Settings → Volumes). Railway's filesystem
  is wiped on every deploy; without a volume the dashboard loses all call
  history each time you push. Mount a second volume at `/app/data` if you want
  customer edits made from the dashboard to survive deploys.

Memory: PyTorch plus the Silero voice-activity model needs roughly 700 MB–1 GB
resident. The Hobby plan (8 GB) is fine; a 512 MB free instance will be killed.

### 1.5 Verify

```bash
curl https://YOUR-RAILWAY-DOMAIN/health
```

In the deploy logs you should see `[STARTUP] Groq models OK`, the Deepgram
connection opening on the first call, and `[VAD] Silero model loaded`.

### 1.6 Twilio console

Outbound calls need no webhook configuration: the backend passes its own TwiML
URL when it creates the call. For **inbound** calls to +1 906 984 8267, set
Phone Numbers → your number → Voice Configuration → "A call comes in" to:

```
POST https://YOUR-RAILWAY-DOMAIN/telephony/twilio/voice
```

While the Twilio account is a trial, calls can only reach verified numbers.

---

## 2. Frontends on Netlify

Two separate Netlify sites from the same repository, each with its own base
directory. `frontend/netlify.toml` and `dashboard/netlify.toml` carry the build
commands; Netlify adds the Next.js runtime plugin automatically.

### Site A — voice frontend

| Setting | Value |
| --- | --- |
| Base directory | `frontend` |
| Build command | `npm run build` (from netlify.toml) |
| Environment | set in `frontend/netlify.toml` |

### Site B — analytics dashboard

| Setting | Value |
| --- | --- |
| Base directory | `dashboard` |
| Build command | `npm run build` (from netlify.toml) |
| Environment | set in `dashboard/netlify.toml` |

**`NEXT_PUBLIC_*` values are baked in at build time.** Set them before the
first build, and trigger a redeploy after any change — editing them without
rebuilding does nothing.

The backend already allows any `https://*.netlify.app` origin. A custom domain
needs adding to `allow_origins` in `app/main.py`.

---

## 3. The "Make a Phone Call" button

On the voice frontend, next to Start/Stop. It sends:

```
POST {backend}/telephony/call/outbound   {}
```

With an empty body the backend dials `TWILIO_OUTBOUND_DEFAULT_TO`, so the
button rings whichever verified number is configured on Railway. The button
reports the number it called, or the backend's error message if Twilio refuses.

---

## Known gaps

- **The dashboard API has no authentication.** Once the backend is public,
  anyone with the URL can read call transcripts and place calls via
  `/api/dashboard/calls/dial`. Before showing it outside a demo, put the
  dashboard routes behind a shared secret header or Railway's private
  networking.
- **The Twilio auth token was shared in chat.** Rotate it in the Twilio console
  after the demo and update the Railway variable.
- `logs/` and `data/` are ephemeral without the volumes described in 1.4.

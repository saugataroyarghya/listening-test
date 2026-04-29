# Simple Listening Demo

## Run with Docker Compose

1. Set your API key in `.env`:

```bash
GROQ_USE_PAID_KEY=false
GROQ_FREE_KEY=your_free_key_here
GROQ_PAID_KEY=your_paid_key_here
```

2. Build and run:

```bash
docker compose up --build
```

3. Check health:

```bash
curl http://localhost:8000/health
```

The API is available at `http://localhost:8000`.

## Run with Docker

```bash
docker build -t simple-listening-demo .
docker run --rm -p 8000:8000 --env-file .env simple-listening-demo
```

## Notes

- First startup can be slow because Whisper model files are downloaded.
- Compose persists model cache in the `model-cache` Docker volume.
- Default container settings are tuned for low-resource CPU demos.

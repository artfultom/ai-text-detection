```shell
docker compose up --build

curl http://localhost:8000/health
# {"status":"ok"}

curl -X POST http://localhost:8000/score \
-H "Content-Type: application/json" \
-d '{"texts": ["This essay was written by a human.", "AI-generated text example."]}'
```

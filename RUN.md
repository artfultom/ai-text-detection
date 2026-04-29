```shell
docker compose up --build

curl http://localhost:8000/health
# {"status":"ok"}

curl -X POST http://localhost:8000/score \
-H "Content-Type: application/json" \
-d '{"texts": ["This essay was written by a human.", "AI-generated text example."]}'
```


```shell
docker build --platform linux/amd64 -t ai-text-detector .
yc container registry list
docker tag ai-text-detector cr.yandex/crpld0k8ub0uc0rem1an/ai-text-detector:latest
docker push cr.yandex/crpld0k8ub0uc0rem1an/ai-text-detector:latest
```


curl -X POST https://bbae3v2b4jj76anecipm.containers.yandexcloud.net/score \
-H "Content-Type: application/json" \
-d '{"texts": ["This essay was written by a human.", "AI-generated text example."]}'
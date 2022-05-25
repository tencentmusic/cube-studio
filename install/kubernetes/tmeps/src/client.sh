curl -d '{"instances": [{"movie_id":1, "user_id":1, "user_rating":0.222}]}' -XPOST http://test-model-server-svc.tfra-dev:8501/v1/models/test/versions/1640092867:predict

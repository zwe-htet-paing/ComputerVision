import sys
from pathlib import Path
from locust import HttpUser, task, between

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from tests.helper import predict_test

class PerformanceTests(HttpUser):
    wait_time = between(1, 3)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.predict_endpoint = "/predict/"

    @task(1)
    def test_fastapi(self):
        response = self.client.get('/')
        print(response.json())

    @task(2)
    def test_predict(self):
        res = self.predict()
        print("res", res)

    def predict(self):
        return predict_test(self.client, self.predict_endpoint)

# Run Locust
# Execute this script using the following command in the terminal:
# locust -f path/to/your/script.py

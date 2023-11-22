import unittest
from fastapi.testclient import TestClient
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
# print(sys.path)

from tests.helper import predict_test
from app.app import app

class TestApp(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_hello(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"msg": "Hello World!!!"})

    def test_predict(self):
        # Include a custom request in test_predict
        custom_request_data = {"img_url": "https://github.com/pytorch/hub/raw/master/images/dog.jpg"}
        response = self.client.post("/predict", json=custom_request_data)
        self.assertEqual(response.status_code, 200)

if __name__ == "__main__":
    

    # Run the tests
    unittest.main()

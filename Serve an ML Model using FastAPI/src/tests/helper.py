import json
from utils.test_images import image_links
from schemas.schemas import ImageRequest


def predict_test(client, api_url):
    sample = ImageRequest(img_url=image_links[0]['url'])
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json'
    }
    res = client.post(api_url,
                      data=json.dumps(sample.dict()),
                      headers=headers
                      )
    return res.json()

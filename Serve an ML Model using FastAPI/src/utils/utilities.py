import os
import requests
from PIL import Image

current_dir = os.getcwd()
# print(current_dir)
os.chdir(current_dir) # change dir

def check_dir(dir_path):
    isExist = os.path.exists(dir_path)
    if not isExist:
        os.makedirs(dir_path)
        print("The new directory is created!")
    else:
        print("The directory already exist!")


def load_image(image_url):
    try:
        img = Image.open(requests.get(image_url, stream=True).raw)
        return img
    except Exception as e:
        print(e)
        print("Image could not be opened...")
        
        
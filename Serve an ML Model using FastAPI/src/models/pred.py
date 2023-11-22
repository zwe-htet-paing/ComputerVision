import torch
from torchvision import models
from torchvision import transforms
import os

# import sys
# sys.path.insert(0, "../utils")
from utils.utilities import *

SAVE_LOCATION = os.getcwd() + "/resources/"

def image_preprocess(img):
    try:
        # Define a series of image transformations
        transform = transforms.Compose([  # [1]
            transforms.Resize(256),  # [2]
            transforms.CenterCrop(224),  # [3]
            transforms.ToTensor(),  # [4]
            transforms.Normalize(  # [5]
                mean=[0.485, 0.456, 0.406],  # [6]
                std=[0.229, 0.224, 0.225]  # [7]
            )])
        input_tensor = transform(img)
        input_batch = torch.unsqueeze(input_tensor, 0)
        return input_batch
    except Exception as e:
        print(e)

def get_model(model=None):
    # global model
    # Check if the model is already loaded, if not, load it
    if model is None:
        model = models.alexnet(pretrained=True)
    return model

def get_labels():
    if not os.path.exists(SAVE_LOCATION + 'imagenet_classes.txt'):
        check_dir(SAVE_LOCATION)
        print("Downloading imagenet class in: ", SAVE_LOCATION)
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        r = requests.get(url, allow_redirects=True)
        open(SAVE_LOCATION + "imagenet_classes.txt", "wb").write(r.content)
    
    # Read the labels
    with open(SAVE_LOCATION + 'imagenet_classes.txt') as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

def predict(input_batch, model=None):
    if model is None:
        model = get_model()
        
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model = model.to('cuda')
    
    # Set the model to evaluation mode 
    model.eval()
    with torch.no_grad():
        output = model(input_batch)
        
    # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
    # print(output.shape)
    
    # Load the labels
    labels = get_labels()
    
    # Get the index with the maximum confidence score
    _, index = torch.max(output, 1)
    
    # Calculate the percentage confidence for the predicted class
    percentage = torch.nn.functional.softmax(output, dim=1)[0] * 100
    
    # Print the predicted label and confidence percentage
    print(labels[index[0]], percentage[index[0]].item())
    
    # Sort the output to get the top predictions
    _, indices = torch.sort(output, descending=True)
    
    # Return the top 5 predictions with labels and confidence percentages
    return [(labels[idx], percentage[idx].item()) for idx in indices[0]][:5]

if __name__ == "__main__":
    image_url = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
    img = load_image(image_url)
    img = image_preprocess(img)
    
    # Make predictions on the input image
    predictions = predict(img)
    print(predictions)
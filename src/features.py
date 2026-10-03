import os
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from src.preprocessing import all_data_filtered


DATA_ROOT = r"C:\Users\Asus\Downloads\TrainingData\TrainingData"

img_folder = os.path.join(DATA_ROOT, "Images")

entry = all_data_filtered[0]

image_id = entry['image_id'] # to be used for extracting images size


image_path = os.path.join(img_folder, f"{image_id}.png")
img = Image.open(image_path)

resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet.eval()
#print(resnet)

feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
feature_extractor.eval()


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])



orig_width, orig_height = img.size

def fixation_to_grid(x, y, orig_width, orig_height, grid_size=7):
    grid_x = int((x / orig_width) * grid_size)
    grid_y = int((y / orig_height) * grid_size)

    grid_x = min(grid_x, grid_size - 1)
    grid_y = min(grid_y, grid_size - 1)
    
    return grid_x, grid_y



images_features_cache = {}

for image_id in range(1, 301):

    image_path = os.path.join(img_folder, f"{image_id}.png")
    img = Image.open(image_path).convert('RGB')
    orig_width, orig_height = img.size

    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
     features = feature_extractor(img_tensor)

     images_features_cache[image_id] = (features[0] , orig_width, orig_height)



def build_feature_sequence(entry,cache):

    features, orig_width, orig_height = cache[entry['image_id']]
    seq = []

    for (x, y, duration) in entry['scanpaths']:
         
         gx, gy = fixation_to_grid(x, y, orig_width, orig_height)
         feat = features[:, gy, gx]
         norm_duration = duration / 1000.0 
         feat_with_duration = torch.cat([feat, torch.tensor([norm_duration], dtype=torch.float32)])
         seq.append(feat_with_duration)

    return torch.stack(seq)
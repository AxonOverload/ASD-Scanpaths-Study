import os
import csv
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import DataLoader
from collections import Counter
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score,recall_score,precision_score,confusion_matrix
import numpy as np
import seaborn as sns



def scanpaths_sparser(filepath):

    scanpaths = []
    current_scanpath = []

    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        next(reader)

        for row in reader:
            idx,x,y,duration = int(row[0]), int(row[1]), int(row[2]), int(row[3])

            if idx==0 and current_scanpath:
                scanpaths.append(current_scanpath)
                current_scanpath = []

            current_scanpath.append((x,y,duration))

        if current_scanpath:
                scanpaths.append(current_scanpath)

        return scanpaths



DATA_ROOT = r"C:\Users\Asus\Downloads\TrainingData\TrainingData"

asd_folder = os.path.join(DATA_ROOT, "ASD")
td_folder = os.path.join(DATA_ROOT, "TD")


all_data = []


for file in sorted(os.listdir(asd_folder)):
     
     filepath = os.path.join(asd_folder, file)
     image_id = int(file.split('_')[-1].split('.')[0])

     scanpaths = scanpaths_sparser(filepath)

     for participant_id, sp in enumerate(scanpaths):
             all_data.append({'participant_id': f'ASD_{participant_id}', 'image_id': image_id,
                               'scanpaths': sp, 'label': 'ASD' })
     

#print(len(all_data))
#print(all_data[0])


for file in sorted(os.listdir(td_folder)):

    filepath = os.path.join(td_folder, file)
    image_id = int(file.split('_')[-1].split('.')[0])

    scanpaths = scanpaths_sparser(filepath)

    for participant_id, sp in enumerate(scanpaths):
        all_data.append({'participant_id': f'TD_{participant_id}', 'image_id': image_id,
                          'scanpaths': sp, 'label': 'TD' })


id_counts = Counter(e['participant_id'] for e in all_data)
valid_ids = {pid for pid, counts in id_counts.items() if counts >=50} # threshold (must appear in at least 50 images)


all_data_filtered = [e for e in all_data if e['participant_id'] in valid_ids]

#print(len(all_data))


DATA_ROOT = r"C:\Users\Asus\Downloads\TrainingData\TrainingData"

img_folder = os.path.join(DATA_ROOT, "Images")

entry = all_data_filtered[0]

image_id = entry['image_id']
scanpath = entry['scanpaths']

image_path = os.path.join(img_folder, f"{image_id}.png")
img = Image.open(image_path)

xs = [point[0] for point in scanpath]
ys = [point[1] for point in scanpath]

plt.figure(figsize=(10,8))
plt.imshow(img)
plt.plot(xs, ys, 'o-', color='red', markersize=8, linewidth=2)
plt.scatter(xs[0], ys[0], color='lime', s=150, zorder=5, label='first fixation')
plt.scatter(xs[-1], ys[-1], color='blue', s=150, zorder=5, label='last fixation')
plt.legend()
plt.title(f"Image {image_id} — {entry['label']} — {len(scanpath)} fixations")
#plt.savefig('asd_detection_plot.png')



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





def mean_pooling(entries,cache):
     
     X = []
     Y = []

     for e in entries:

          seq = build_feature_sequence(e,cache)
          pooled = seq.mean(dim=0)

          X.append(pooled.numpy())
          Y.append(1 if e['label'] =='ASD' else 0)

     return np.array(X), np.array(Y)




from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
sgkf = StratifiedGroupKFold(n_splits=7)


groups = [e['participant_id'] for e in all_data_filtered]
labels = [1 if e['label']=='ASD' else 0 for e in all_data_filtered]

scaler = StandardScaler()

def pooled_cv():
     accs=[]

     for fold, (train_idx, test_idx) in enumerate(sgkf.split(all_data_filtered, labels, groups=groups)):
        train_entries = [all_data_filtered[i] for i in train_idx]
        test_entries = [all_data_filtered[i] for i in test_idx]
        
        X_train, y_train = mean_pooling(train_entries, images_features_cache)
        X_test, y_test = mean_pooling(test_entries, images_features_cache)

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train,y_train)
        preds = model.predict(X_test)
        
        acc = (preds == y_test).mean()
        print(f"Fold {fold}: acc={acc:.4f}, unique predictions={set(preds)}, test class balance={y_test.mean():.3f}")
        accs.append(acc)


     print(f"Mean: {np.mean(accs):.4f},{np.std(accs):.4f}")
     return accs


pooled_cv()
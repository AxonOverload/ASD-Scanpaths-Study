import os
import csv
from collections import Counter



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

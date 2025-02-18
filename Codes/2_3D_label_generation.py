import yaml
import numpy as np
import torch
from pathlib import Path
import torchio as tio
import pandas as pd
from utils.processing_masks import pred_to_segment_3ch, prediction_mask_computation
from models.CNN_3D import Unsupervised_Segmentation_Model
from utils.data_utils import process_subjects

# Load configuration
with open("../config.yaml", "r") as file:
    config = yaml.safe_load(file)

device_ids = [config["device"]]
device = torch.device("cuda:%d" % device_ids[0] if torch.cuda.is_available() else "cpu")
print(device)

####### DATASET ########
print('Preparing dataset...')
df = pd.read_excel(config["path_to_excel"])

h, w, depth = config["h"], config["w"], config["depth"]

# Modality 1 - Original Volume
val_list_T1 = df.loc[((df["data_split"] == config['lab_generation_split']) & (df["modality"] == config["modality_1"]))]
val_list_T1 = val_list_T1["FILE"]
val_paths_T1 = [Path(i) for i in val_list_T1]

# Modality 2 - Original Volume
val_list_T2 = df.loc[((df["data_split"] == config['lab_generation_split']) & (df["modality"] == config["modality_2"]))]
val_list_T2 = val_list_T2["FILE"]
val_paths_T2 = [Path(i) for i in val_list_T2]

# Modality 1 - Label
val_labels_T1 = df.loc[((df["data_split"] == config['lab_generation_split']) & (df["modality"] == config["modality_1_label"]))]
val_labels_T1 = val_labels_T1['FILE']
val_paths_labels_T1 = [Path(i) for i in val_labels_T1]

# Modality 2 - Label
val_labels_T2 = df.loc[((df['data_split'] == config['lab_generation_split']) & (df['modality'] == config["modality_2_label"]))]
val_labels_T2 = val_labels_T2['FILE']
val_paths_labels_T2 = [Path(i) for i in val_labels_T2]


val_sbj_T2 = process_subjects(val_paths_T2, h, w, depth)
val_sbj_T1 = process_subjects(val_paths_T1, h, w, depth)

val_labels_T2 = process_subjects(val_paths_labels_T2, h, w, depth)
val_labels_T1 = process_subjects(val_paths_labels_T1, h, w, depth)

batch_size = config["batch_size"]
num_workers = config["num_workers"]

val_dataset_T2 = tio.SubjectsDataset(val_sbj_T2)
val_dataset_T1 = tio.SubjectsDataset(val_sbj_T1)

val_loader_T2 = torch.utils.data.DataLoader(val_dataset_T2, batch_size=batch_size,
                                         num_workers=num_workers, shuffle=False,
                                            pin_memory=True)

val_loader_T1 = torch.utils.data.DataLoader(val_dataset_T1, batch_size=batch_size,
                                         num_workers=num_workers,shuffle=False,
                                            pin_memory=True)

######## Model ######
# Load Model
input_channels = config["input_channels"]
nConv = config["nConv"]
nChannel = config["nChannel"]
model = Unsupervised_Segmentation_Model(input_channels, nConv=nConv, nChannel=nChannel)
model.load_state_dict(torch.load("../Pretrained_3DCNN/best_model.pt"))
model.to(device)

###### Label Generation ######
# (1) Predict and Compute Unsupervised Segmentations:
print('Predicting unsupervised segmentation...')
predictions = []      
model.eval()
with torch.no_grad():
    for batch1, batch2 in zip(val_loader_T2, val_loader_T1):
        val_batch_T2 = batch1['MRI']['data'].to(device)
        val_batch_T1 = batch2['MRI']['data'].to(device)

        # Forward
        pred_batch = model(val_batch_T2, val_batch_T1).to(device)
        predictions.append(pred_batch)

label_colours = np.array([[21, 239, 150],[182, 6, 195], [48, 40, 234],
                [228, 148, 238],[25, 204, 198],[177, 214, 37],[59, 167, 56],
                [72, 206, 119],[86, 234, 104],[199, 214, 249],[123, 42, 87],
                [134, 113, 155],[198, 53, 135],[46, 128, 103],[90, 84, 34],
                [242, 252, 156],[233, 150, 210],[225, 203, 175],[215, 46, 80],
                [78, 93, 47],[211, 57, 211],[238, 113, 6],[22, 63, 207],
                [171, 34, 60],[65, 68, 47],[155, 66, 99],[218, 131, 85],
                [130, 218, 147],[234, 115, 224],[24, 114, 22]])
print('Computing unsupervised segmentation...')
segmentations = []
for batch in range(len(predictions)):
    batch1 = predictions[batch]
    for sbj in range(batch1.shape[0]):
        subject = batch1[sbj,:,:,:,:]
        sbj_volume = pred_to_segment_3ch(subject, nChannel, label_colours, depth)
        segmentations.append(sbj_volume)
        
        
# (2) Set seeds and compute AI assisted labels
print("Generating AI assisted labels...")
generated_labels = []
num = len(segmentations)
for i in range(num):
    pred_labels = []
    for s in range(depth):
        sbj = segmentations[i][s, :, :, :]
        labT1 = val_labels_T1[i]['MRI'].numpy()[:, :, :, s].transpose(1, 2, 0)
        labT2 = val_labels_T2[i]['MRI'].numpy()[:, :, :, s].transpose(1, 2, 0)
        
        # Compute label
        mask = labT1+labT2
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1

        slice_mask = prediction_mask_computation(sbj.cpu().numpy(),mask, min_thresh=0, max_thresh=2)
        pred_labels.append(slice_mask)
        
    generated_labels.append(np.array(pred_labels))
    
# Save
for i in range(len(segmentations)):
    np.savez(f'../Generated_Labels/label_generated_case{i}.npz', Unsupervised_seg=segmentations[i], AI_Label=generated_labels[i])
print('AI assisted labels saved in Generated_Labels/.')


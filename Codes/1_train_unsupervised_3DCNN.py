import yaml
import numpy as np
import torch
import torch.optim as optim
import torch.nn.init
from pathlib import Path
import torchio as tio
import pandas as pd
from models.CNN_3D import Unsupervised_Segmentation_Model, myloss3D_opt
from utils.data_utils import process_subjects
import warnings
warnings.simplefilter("ignore")

# Load configuration
with open("../config.yaml", "r") as file:
    config = yaml.safe_load(file)

device_ids = [config["device"]]
device = torch.device("cuda:%d" % device_ids[0] if torch.cuda.is_available() else "cpu")
print(device)

####### DATASET ########
df = pd.read_excel(config["path_to_excel"])

h, w, depth = config["h"], config["w"], config["depth"]

# Modality 1
train_list_T1 = df.loc[((df["data_split"] == "TRAIN") & (df["modality"] == config["modality_1"]))]
train_list_T1 = train_list_T1["FILE"]
val_list_T1 = df.loc[((df["data_split"] == "VAL") & (df["modality"] == config["modality_1"]))]
val_list_T1 = val_list_T1["FILE"]
train_paths_T1 = [Path(i) for i in train_list_T1]
val_paths_T1 = [Path(i) for i in val_list_T1]

# Modality 2
train_list_T2 = df.loc[((df["data_split"] == "TRAIN") & (df["modality"] == config["modality_2"]))]
train_list_T2 = train_list_T2["FILE"]
val_list_T2 = df.loc[((df["data_split"] == "VAL") & (df["modality"] == config["modality_2"]))]
val_list_T2 = val_list_T2["FILE"]
train_paths_T2 = [Path(i) for i in train_list_T2]
val_paths_T2 = [Path(i) for i in val_list_T2]


train_sbj_T2 = process_subjects(train_paths_T2, h, w, depth)
val_sbj_T2 = process_subjects(val_paths_T2, h, w, depth)
train_sbj_T1 = process_subjects(train_paths_T1, h, w, depth)
val_sbj_T1 = process_subjects(val_paths_T1, h, w, depth)
    

train_transform = tio.Compose([tio.RandomFlip(axes=('LR',), flip_probability = 0.3),
tio.RandomAffine(scales=(0.9, 1.2),degrees=15)])

train_dataset_T2 = tio.SubjectsDataset(train_sbj_T2, transform=train_transform)
val_dataset_T2 = tio.SubjectsDataset(val_sbj_T2)

print(f"There are {len(train_sbj_T2)} train subjects and {len(val_sbj_T2)} val subjects in Modality 1")

train_dataset_T1 = tio.SubjectsDataset(train_sbj_T1, transform=train_transform)
val_dataset_T1 = tio.SubjectsDataset(val_sbj_T1)

print(f"There are {len(train_sbj_T1)} train subjects and {len(val_sbj_T1)} val subjects in Modality 2")


batch_size = config["batch_size"]
num_workers = config["num_workers"]
train_loader_T2 = torch.utils.data.DataLoader(train_dataset_T2, batch_size=batch_size,
                                           num_workers=num_workers, shuffle=False,
                                              pin_memory=True)
val_loader_T2 = torch.utils.data.DataLoader(val_dataset_T2, batch_size=batch_size,
                                         num_workers=num_workers, shuffle=False,
                                            pin_memory=True)

train_loader_T1 = torch.utils.data.DataLoader(train_dataset_T1, batch_size=batch_size,
                                           num_workers=num_workers, shuffle=False,
                                              pin_memory=True)
val_loader_T1 = torch.utils.data.DataLoader(val_dataset_T1, batch_size=batch_size,
                                         num_workers=num_workers,shuffle=False,
                                            pin_memory=True)
###### MODEL #######
input_channels = config["input_channels"]
nConv = config["nConv"]
nChannel = config["nChannel"]
model = Unsupervised_Segmentation_Model(input_channels, nConv=nConv, nChannel=nChannel)
model.to(device)

###### TRAINING #####
lr = config["lr"]
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

num_epochs = config["num_epochs"]
mean_train_loss = []
mean_val_loss = []
best_val_loss = np.Inf
epochs_since_improvement = 0
patience = config["patience"]
converge = 0
for epoch in range(num_epochs):
    model.train()
    batch_idx=1
    epoch_train_loss = []
    epoch_val_loss = []
    for batch1, batch2 in zip(train_loader_T2, train_loader_T1):
        train_batch_T2 = batch1['MRI']['data'].to(device)
        train_batch_T1 = batch2['MRI']['data'].to(device)

        # Forward
        optimizer.zero_grad()
        pred_batch = model(train_batch_T2, train_batch_T1).to(device)
        torch.cuda.empty_cache()

        # Loss
        loss, nLabels = myloss3D_opt(pred_batch,train_batch_T2, nChannel*2, device)
        torch.cuda.empty_cache()

        # Backpropagation
        loss.backward()

        # Optimization step
        optimizer.step()

        epoch_train_loss.append(loss.item())
        print ('Epoch:',epoch,"| batch:", batch_idx,' | loss :', loss.item(), "| N Clusters: ", nLabels)
        batch_idx = batch_idx+1
        torch.cuda.empty_cache()

    mean_train_loss.append(np.mean(epoch_train_loss))

    # VALIDATION
    model.eval()
    with torch.no_grad():
        for batch1, batch2 in zip(val_loader_T2, val_loader_T1):
            val_batch_T2 = batch1['MRI']['data'].to(device)
            val_batch_T1 = batch2['MRI']['data'].to(device)

            # Forward
            pred_batch = model(val_batch_T2, val_batch_T1).to(device)

            # Loss
            loss, nLabels = myloss3D_opt(pred_batch,val_batch_T2, nChannel*2, device)

            epoch_val_loss.append(loss.item())
            del val_batch_T2, val_batch_T1, pred_batch, loss
            torch.cuda.empty_cache()

    mean_val_loss.append(np.mean(epoch_val_loss))
    print("Val Loss: ", np.mean(epoch_val_loss))

    # Check if the validation loss has improved
    if  np.mean(epoch_val_loss) < best_val_loss:
        best_val_loss = np.mean(epoch_val_loss)
        epochs_since_improvement = 0
        converge = epoch
        torch.save(model.state_dict(), '../Pretrained_3DCNN/best_model.pt')
    else:
        epochs_since_improvement += 1

    # Check if training should be stopped
    if epochs_since_improvement >= patience:
        print('Training stopped because validation loss did not improve for {} epochs.'.format(patience))
        np.save('../Pretrained_3DCNN/train_losses.npy', mean_train_loss)
        np.save('../Pretrained_3DCNN/val_losses.npy', mean_val_loss)
        break

    if epoch == (num_epochs-1):
        print("Reached max num of epochs.")
        np.save('../Pretrained_3DCNN/train_losses.npy', mean_train_loss)
        np.save('../Pretrained_3DCNN/val_losses.npy', mean_val_loss)
        break
# Semi-Supervised Label Generation for 3D Multi-Modal MRI Bone Tumor Segmentation

**Authors:**  
Anna Curto-Vilalta, Benjamin Schlossmacher, Christina Valle, Alexandra Gersing, Jan Neumann, Ruediger von Eisenhart-Rothe, Daniel Rueckert, and Florian Hinterwimmer  

### Overview  
This framework generates AI-assisted labels as segmentation annotations for supervised segmentation tasks. It is the official implementation of the paper:  
**"Semi-Supervised Label Generation for 3D Multi-Modal MRI Bone Tumor Segmentation"**  

The framework is adaptable to any volumetric dataset, with the only requirement being the presence of two image modalities.  

### Installation  
1. Clone the Git repository:  
   ```bash
   git clone https://github.com/acurtovilalta/3D_LabelGeneration.git
   ```
2. Create a Conda environment:  
   ```bash
   conda create --name lab_gen_env python=3.8
   conda activate lab_gen_env
   ```
3. Install the required dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

### Step 1: Data Preparation  
You need an Excel file containing three columns that list all data samples for the two MRI modalities alongside their annotation labels.  

##### Required Columns:
- `FILE`: Path to the `.nii.gz` file.  
- `modality`: MRI modality (e.g., `T1`, `T2`, `T1_label`, `T2_label`).  
- `data_split`: Data split category (`TRAIN`, `VAL`, `TEST`).  

##### Preprocessing
The script automatically performs: Resizing and Z-normalization. For large datasets, preprocessing can be skipped to save time and resources by storing already processed images.  

### Step 2: Training the Unsupervised 3D CNN  
1. Modify the `config.yaml` file according to your requirements.  
2. Run the training script:  
   ```bash
   python 1_train_unsupervised_3DCNN.py
   ```
   - Once training converges, the best model's state dictionary will be saved in the `Pretrained_3DCNN` folder, along with training and validation losses.  

### Step 3: Computing AI-Assisted Labels  
1. Modify the `config.yaml` file to specify the data split for AI-assisted label generation.  
2. Execute the label generation script:  
   ```bash
   python 2_3D_label_generation.py
   ```
   - This will generate `.npz` files for each sample in the folder Generated_Labels, containing:  
     - Unsupervised segmentation (`key: Unsupervised_seg`)  
     - AI-assisted label (`key: AI_Label`)  

### Citing  
If you use this repository in your work, please consider citing:  

```bibtex
@misc{3DLabelGen,
  title     = {Semi-Supervised Label Generation for 3D Multi-Modal MRI Bone Tumor Segmentation}, 
  author    = {Anna Curto-Vilalta and Benjamin Schlossmacher and Christina Valle and Alexandra Gersing and Jan Neumann and Ruediger von Eisenhart-Rothe and Daniel Rueckert and Florian Hinterwimmer},
  year      = {2025},
  journal   = {Journal of Imaging Informatics in Medicine}
}
```
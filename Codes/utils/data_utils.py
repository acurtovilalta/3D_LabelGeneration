import numpy as np
import torchio as tio
from pathlib import Path

def volume_processing(mri_volume, h, w, d):
    norm = tio.ZNormalization()
    s_max = np.max(mri_volume.shape)
    square_padding = tio.CropOrPad((s_max,s_max, d))
    mri_pad = square_padding(mri_volume)
    resize_proc = tio.Resize((h, w, d))
    mri_process = resize_proc(mri_pad)
    mri_tensor = norm(mri_process)
    return(mri_tensor)

def process_subjects(paths, h, w, depth):
    subjects = []
    for subject_path in paths:
        subject = tio.Subject({"MRI": tio.ScalarImage(subject_path)})
        mri_data = subject["MRI"]
        processed_mri_data = volume_processing(mri_data, h, w, depth)
        processed_subject = tio.Subject({"MRI": processed_mri_data})
        subjects.append(processed_subject)
    return subjects

import os
import shutil
import tqdm
import numpy as np
import SimpleITK as sitk
from PIL import Image
from collections import OrderedDict
from batchgenerators.utilities.file_and_folder_operations import *

patient_names = []

out_path = "./Task061_VSseg/"
data_path = "./2021.Vestibular-Schwannoma-SEG/VS-SEG/"

if os.path.exists(out_path):
    shutil.rmtree(out_path)
os.makedirs(out_path)
out_path_image = os.path.join(out_path, "imagesTr")
out_path_label = os.path.join(out_path, "labelsTr")

os.makedirs(out_path_image, exist_ok=True)
os.makedirs(out_path_label, exist_ok=True)
os.makedirs(os.path.join(out_path, "imagesTs"), exist_ok=True)



for sample in tqdm.tqdm(sorted(os.listdir(data_path))):
    if ".DS_Store" in sample or not os.path.isdir(os.path.join(data_path, sample)):
        continue
    patient_names.append(sample)
    img = os.path.join(data_path, sample, "vs_gk_t1_refT1.nii.gz")
    seg = os.path.join(data_path, sample, "vs_gk_seg_refT1.nii.gz")

    shutil.copy(img, os.path.join(out_path_image, sample + "_0000.nii.gz"))
    shutil.copy(seg, os.path.join(out_path_label, sample + ".nii.gz"))

json_dict = OrderedDict()
json_dict['name'] = "Vestibular-Schwannoma-SEG"
json_dict['description'] = "nothing"
json_dict['tensorImageSize'] = "4D"
json_dict['reference'] = "see https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70229053"
json_dict['licence'] = "see Vestibular-Schwannoma-SEG"
json_dict['release'] = "0.0"
json_dict['modality'] = {
    "0": "T1"
}
json_dict['labels'] = {
    "0": "background",
    "1": "tumor"
}
json_dict['numTraining'] = len(patient_names)
json_dict['numTest'] = 0
json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in
                         patient_names]
json_dict['test'] = []

save_json(json_dict, join(out_path, "dataset.json"))

import random
random.seed(1234)
random.shuffle(patient_names)
print("shuffle!")
data_split_train_val = OrderedDict()
with open(os.path.join(out_path, "splits_final.pkl"), "wb") as pk:
    data_split_train_val["train"] = patient_names[:int(0.8*len(patient_names))]
    data_split_train_val["val"] = patient_names[int(0.8*len(patient_names)):]
    pickle.dump([data_split_train_val], pk)
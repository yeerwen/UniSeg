import os
import shutil
import tqdm
import numpy as np
import SimpleITK as sitk
from PIL import Image
from collections import OrderedDict
from batchgenerators.utilities.file_and_folder_operations import *

patient_names = []

out_path = "./Task060_BTCV/"
data_path = "/media/userdisk0/ywye/nnUNet_raw/nnUNet_raw_data/2015.BTCV/RawData/Training/"

if os.path.exists(out_path):
    shutil.rmtree(out_path)
os.makedirs(out_path)
out_path_image = os.path.join(out_path, "imagesTr")
out_path_label = os.path.join(out_path, "labelsTr")

os.makedirs(out_path_image, exist_ok=True)
os.makedirs(out_path_label, exist_ok=True)
os.makedirs(os.path.join(out_path, "imagesTs"), exist_ok=True)



for sample in tqdm.tqdm(sorted(os.listdir(os.path.join(data_path, "img")))):
    if ".DS_Store" in sample:
        continue
    sample_name = "bcv_" + str(int(sample[3:-7]))
    patient_names.append(sample_name)
    img = os.path.join(data_path, "img", sample)
    seg = os.path.join(data_path, "label", sample.replace("img", "label"))

    shutil.copy(img, os.path.join(out_path_image, sample_name + "_0000.nii.gz"))
    shutil.copy(seg, os.path.join(out_path_label, sample_name + ".nii.gz"))

json_dict = OrderedDict()
json_dict['name'] = "BTCV"
json_dict['description'] = "nothing"
json_dict['tensorImageSize'] = "4D"
json_dict['reference'] = "see https://www.synapse.org/#!Synapse:syn3193805/wiki/217789"
json_dict['licence'] = "see BTCV"
json_dict['release'] = "0.0"
json_dict['modality'] = {
    "0": "CT"
}
json_dict['labels'] = {
    "0": "background",
    "1": "spleen",
    "2": "right kidney",
    "3": "left kidney",
    "4": "gallbladder",
    "5": "esophagus",
    "6": "liver",
    "7": "stomach",
    "8": "aorta",
    "9": "inferior vena cava",
    "10": "portal vein and splenic vein",
    "11": "pancreas",
    "12": "right adrenal gland",
    "13": "left adrenal gland"
}
json_dict['numTraining'] = len(patient_names)
json_dict['numTest'] = 0
json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in
                         patient_names]
json_dict['test'] = []

save_json(json_dict, join(out_path, "dataset.json"))

data_split_train_val = OrderedDict()
with open(os.path.join(out_path, "splits_final.pkl"), "wb") as pk:
    data_split_train_val["train"] = ['bcv_1', 'bcv_2', 'bcv_3', 'bcv_4', 'bcv_5', 'bcv_6', 'bcv_7', 'bcv_8', 'bcv_9', 'bcv_10', 'bcv_21', 'bcv_22', 'bcv_23', 'bcv_24', 'bcv_25', 'bcv_26', 'bcv_27', 'bcv_28', 'bcv_29', 'bcv_30', 'bcv_31']
    data_split_train_val["val"] = ['bcv_32', 'bcv_33', 'bcv_34', 'bcv_35', 'bcv_36', 'bcv_37', 'bcv_38', 'bcv_39', 'bcv_40']
    pickle.dump([data_split_train_val], pk)
    print(data_split_train_val)
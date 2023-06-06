import os
import shutil
from batchgenerators.utilities.file_and_folder_operations import *
from collections import OrderedDict
from tqdm import tqdm
data_path = "/media/userdisk1/yeyiwen/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/"
sub_dataset_name = ["Task30_Liver", "Task31_Kidney", "Task32_HepaticVessel", "Task33_Pancreas", "Task34_Colon", "Task35_Lung", "Task36_Spleen"]
out_path = "./Task091_MOTS/"
if os.path.exists(out_path):
    shutil.rmtree(out_path)
os.makedirs(out_path)
out_image_path = os.path.join(out_path, "imagesTr")
out_label_path = os.path.join(out_path, "labelsTr")
os.makedirs(out_image_path)
os.makedirs(out_label_path)
os.makedirs(os.path.join(out_path, "imagesTs"))

patient_names = []
for dataset in sub_dataset_name:
    sub_dataset_image_path = os.path.join(data_path, dataset, "imagesTr")
    sub_dataset_label_path = os.path.join(data_path, dataset, "labelsTr")
    for name in tqdm(os.listdir(sub_dataset_image_path)):
        if name.endswith(".nii.gz"):
            shutil.copy(os.path.join(sub_dataset_image_path, name), os.path.join(out_image_path, name.replace(".nii.gz", "_0000.nii.gz")))
            shutil.copy(os.path.join(sub_dataset_label_path, name), os.path.join(out_label_path, name))
            patient_names.append(name.replace(".nii.gz", ""))


json_dict = OrderedDict()
json_dict['name'] = "MOTS benchmark for multi-organ and tumor segmentation"
json_dict['description'] = "nothing"
json_dict['tensorImageSize'] = "4D"
json_dict['reference'] = "see https://github.com/jianpengz/DoDNet"
json_dict['licence'] = "see https://github.com/jianpengz/DoDNet"
json_dict['release'] = "0.0"
json_dict['modality'] = {
    "0": "CT"
}
json_dict['labels'] = {
    "0": "background",
    "1": "organ",
    "2": "tumor"
}
json_dict['numTraining'] = len(patient_names)
json_dict['numTest'] = 0
json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in
                         patient_names]
json_dict['test'] = []

save_json(json_dict, join(out_path, "dataset.json"))


#we provide data split following DoDNet
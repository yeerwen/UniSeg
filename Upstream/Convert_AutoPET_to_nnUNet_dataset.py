import os
import shutil
import tqdm
import numpy as np
import SimpleITK as sitk
from PIL import Image
from collections import OrderedDict
from batchgenerators.utilities.file_and_folder_operations import *
def NiiDataRead(path, as_type=np.float32):
    img = sitk.ReadImage(path)
    spacing = img.GetSpacing()
    origin = img.GetOrigin()
    direction = img.GetDirection()
    img_it = sitk.GetArrayFromImage(img).astype(as_type)
    return img_it, spacing, origin, direction

def NiiDataWrite(path, prediction_final, spacing, origin, direction):
    # prediction_final = prediction_final.astype(as_type)
    img = sitk.GetImageFromArray(prediction_final)
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    img.SetDirection(direction)
    sitk.WriteImage(img, path)


patient_names = []

out_path = "./Task011_AutoPET/"
data_path = "./2022.AutoPET/FDG-PET-CT-Lesions/"
os.makedirs(out_path, exist_ok=True)
out_path_image = os.path.join(out_path, "imagesTr")
out_path_label = os.path.join(out_path, "labelsTr")

os.makedirs(out_path_image, exist_ok=True)
os.makedirs(out_path_label, exist_ok=True)
os.makedirs(os.path.join(out_path, "imagesTs"), exist_ok=True)


for sample in tqdm.tqdm(sorted(os.listdir(data_path))):
    sample_dir = os.listdir(os.path.join(data_path, sample))
    for itr, sub_sample in enumerate(sample_dir):
        sub_name = sample + "_" + str(itr)
        if ".DS_Store" in sub_sample:
            continue

        CT_res_path = os.path.join(data_path, sample, sub_sample, "CTres.nii.gz")
        SUV_path = os.path.join(data_path, sample, sub_sample, "SUV.nii.gz")
        seg_path = os.path.join(data_path, sample, sub_sample, "SEG.nii.gz")

        seg, spacing_label, origin_label, direction_label = NiiDataRead(seg_path)
        if np.max(seg) <= 0:
            continue

        shutil.copy(CT_res_path, os.path.join(out_path_image, sub_name + "_0000.nii.gz"))
        shutil.copy(SUV_path, os.path.join(out_path_image, sub_name + "_0001.nii.gz"))
        NiiDataWrite(os.path.join(out_path_label, sub_name + ".nii.gz"), seg, spacing_label, origin_label, direction_label)
        patient_names.append(sub_name)



json_dict = OrderedDict()
json_dict['name'] = "autoPET2022"
json_dict['description'] = "nothing"
json_dict['tensorImageSize'] = "4D"
json_dict['reference'] = "see autoPET2022"
json_dict['licence'] = "see autoPET2022 license"
json_dict['release'] = "0.0"
json_dict['modality'] = {
    "0": "CT",
    "1": "PET",
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
    print(len(data_split_train_val["train"]), len(data_split_train_val["val"]))
    pickle.dump([data_split_train_val], pk)
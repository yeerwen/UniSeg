import shutil
import os
import json
import numpy as np
import SimpleITK as sitk
from collections import OrderedDict
import pickle
def NiiDataRead(path, as_type=np.float32):
    img = sitk.ReadImage(path)
    spacing = img.GetSpacing()
    origin = img.GetOrigin()
    direction = img.GetDirection()
    img_it = sitk.GetArrayFromImage(img).astype(as_type)
    return img_it, spacing, origin, direction

def NiiDataWrite(path, prediction_final, spacing, origin, direction):
    img = sitk.GetImageFromArray(prediction_final, isVector=False)
    # print(img.GetSize())
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    img.SetDirection(direction)
    sitk.WriteImage(img, path)

data_path = "/data/userdisk0/ywye/nnUNet_raw/nnUNet_raw_data/Processed_data_nii/"
out_path = "./Task020_Prostate/"
if os.path.exists(out_path):
    shutil.rmtree(out_path)
os.makedirs(out_path)
image_path = os.path.join(out_path, "imagesTr")
label_path = os.path.join(out_path, "labelsTr")
os.makedirs(image_path)
os.makedirs(label_path)
os.makedirs(os.path.join(out_path, "imagesTs"))

index = 0
sub_dataset = ["BIDMC", "BMC", "HK", "I2CVB", "RUNMC", "UCL"]

training_data_name, val_data_name  = [], []
for sub_dataset_name in sub_dataset:
    print(sub_dataset_name)
    mr_nii_name = sorted(os.listdir(os.path.join(data_path, sub_dataset_name)))
    assert len(mr_nii_name) % 2 == 0
    training_name = mr_nii_name[:int(0.8*len(mr_nii_name)//2)*2]
    val_name = mr_nii_name[int(0.8*len(mr_nii_name)//2)*2:]
    assert len(training_name) % 2 == 0 and len(val_name) % 2 == 0, [len(training_name), len(val_name)]
    for seq in range(0, len(training_name), 2):
        assert training_name[seq].split(".")[0] == training_name[seq+1].split("_")[0]
        seg, spacing_1, origin_1, direction_1 = NiiDataRead(os.path.join(data_path, sub_dataset_name, training_name[seq+1]))
        image, spacing_2, origin_2, direction_2 = NiiDataRead(os.path.join(data_path, sub_dataset_name, training_name[seq]))
        assert seg.shape == image.shape, [seg.shape, image.shape]

        shutil.copyfile(os.path.join(data_path, sub_dataset_name, training_name[seq]), os.path.join(image_path, "prostate_"+str(index)+"_0000.nii.gz"))
        seg[seg>0] = 1

        print(training_name[seq].split(".")[0], seg.shape, np.min(seg), np.max(seg))
        NiiDataWrite(os.path.join(label_path, "prostate_"+str(index)+".nii.gz"), seg, spacing_2, origin_2, direction_2)
        training_data_name.append("prostate_"+str(index)+".nii.gz")
        index += 1
    for seq in range(0, len(val_name), 2):
        assert val_name[seq].split(".")[0] == val_name[seq+1].split("_")[0]
        seg, spacing_1, origin_1, direction_1 = NiiDataRead(os.path.join(data_path, sub_dataset_name, val_name[seq+1]))
        image, spacing_2, origin_2, direction_2 = NiiDataRead(os.path.join(data_path, sub_dataset_name, val_name[seq]))
        assert seg.shape == image.shape, [seg.shape, image.shape]

        shutil.copyfile(os.path.join(data_path, sub_dataset_name, val_name[seq]), os.path.join(image_path, "prostate_"+str(index)+"_0000.nii.gz"))
        seg[seg>0] = 1
        print(val_name[seq].split(".")[0], seg.shape, np.min(seg), np.max(seg))
        NiiDataWrite(os.path.join(label_path, "prostate_"+str(index)+".nii.gz"), seg, spacing_2, origin_2, direction_2)
        val_data_name.append("prostate_"+str(index)+".nii.gz")
        index += 1
print("training:", len(training_data_name), "test:", len(val_data_name))
print(training_data_name)
print(val_data_name)
# print(config)
dataset_info = {}
dataset_info["name"] = "mutli-center Prostate Segmentation"
dataset_info["description"] = "collecing from BIDMC, BMC, HK, I2CVB, RUNMC, and UCL datasets"
dataset_info["reference"] = "https://liuquande.github.io/SAML/"
dataset_info["licence"] = "GNU General Public License v3.0"
dataset_info["release"] = ""
dataset_info["tensorImageSize"] = "4D"
dataset_info["modality"] = {"0": "T2"}
dataset_info["labels"] = {"0": "background", "1": "prostate"}
dataset_info["numTraining"] = len(training_data_name)+len(val_data_name)
dataset_info["numTest"] = 0

train_list, test_list = [], []
for name in training_data_name:
    train_list.append({"image": "./imagesTr/"+name, "label": "./labelsTr/"+name})
for name in val_data_name:
    train_list.append({"image": "./imagesTr/"+name, "label": "./labelsTr/"+name})


dataset_info["training"] = train_list
dataset_info["test"] = test_list

with open(os.path.join(out_path, "dataset.json"), 'w', encoding='utf-8') as fw:
    json.dump(dataset_info, fw)

data_split_train_val = OrderedDict()
training_list = training_data_name
testing_list = val_data_name
training_list = [k.replace(".nii.gz","") for k in training_list]
testing_list = [k.replace(".nii.gz","") for k in testing_list]
assert  len(training_list) == 91 and len(testing_list) == 25
with open(os.path.join(out_path, "splits_final.pkl"), "wb") as pk:
    data_split_train_val["train"] = training_list
    data_split_train_val["val"] = testing_list
    pickle.dump([data_split_train_val], pk)

import os
import shutil

path = "/media/userdisk1/yeyiwen/nnUNetFrame/DATASET/nnUNet_raw_data_base/nnUNet_raw_data/1Kidney/origin/"
output = "/media/userdisk1/yeyiwen/nnUNetFrame/DATASET/nnUNet_raw_data_base/nnUNet_raw_data/1Kidney/"
imagesTr = os.path.join(output, "imagesTr")
labelsTr = os.path.join(output, "labelsTr")
os.makedirs(imagesTr)
os.makedirs(labelsTr)
for case in os.listdir(path):
    patient_path = os.path.join(path, case)
    shutil.copyfile(os.path.join(patient_path, "imaging.nii.gz"), os.path.join(imagesTr, "kidney_{}.nii.gz".format(int(case.split("_")[-1]))))
    shutil.copyfile(os.path.join(patient_path, "segmentation.nii.gz"),
                    os.path.join(labelsTr, "kidney_{}.nii.gz".format(int(case.split("_")[-1]))))


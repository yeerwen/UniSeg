"""
https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/dataset_conversion/Task083_VerSe2020.py
"""

import shutil
import os
import json
import numpy as np
import SimpleITK as sitk
from collections import OrderedDict
import pickle
from tqdm import tqdm
from batchgenerators.utilities.file_and_folder_operations import *
from multiprocessing import Pool
import nibabel as nib
from nibabel import io_orientation


def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):

    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()
    originSpacing = itkimage.GetSpacing()
    newSize = np.array(newSize,float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(np.int)
    resampler.SetReferenceImage(itkimage)
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)
    return itkimgResampled


def NiiDataRead(path, as_type=np.float32):
    img = sitk.ReadImage(path)
    spacing = img.GetSpacing()
    origin = img.GetOrigin()
    direction = img.GetDirection()
    # img_it = sitk.GetArrayFromImage(img).astype(np.float32)
    return img, spacing, origin, direction

def NiiDataWrite(path, prediction_final, spacing, origin, direction):
    img = sitk.GetImageFromArray(prediction_final)
    # print(img.GetSize())
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    img.SetDirection(direction)
    sitk.WriteImage(img, path)

def NiiDataWrite_itk(path, prediction_final, spacing, origin, direction):
    img = prediction_final
    # print(img.GetSize())
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    img.SetDirection(direction)
    sitk.WriteImage(img, path)

def reorient_to_ras(image: str) -> None:
    """
    Will overwrite image!!!
    :param image:
    :return:
    """
    assert image.endswith('.nii.gz')
    origaffine_pkl = image[:-7] + '_originalAffine.pkl'
    if not isfile(origaffine_pkl):
        img = nib.load(image)
        original_affine = img.affine
        original_axcode = nib.aff2axcodes(img.affine)
        img = img.as_reoriented(io_orientation(img.affine))
        new_axcode = nib.aff2axcodes(img.affine)
        print(image.split('/')[-1], 'original axcode', original_axcode, 'now (should be ras)', new_axcode)
        nib.save(img, image)
        save_pickle((original_affine, original_axcode), origaffine_pkl)

def reorient_all_images_in_folder_to_ras(folder: str, num_processes: int = 8):
    p = Pool(num_processes)
    nii_files = subfiles(folder, suffix='.nii.gz', join=True)
    p.map(reorient_to_ras, nii_files)
    p.close()
    p.join()

def check_if_all_in_good_orientation(imagesTr_folder: str, labelsTr_folder: str, output_folder: str) -> None:
    maybe_mkdir_p(output_folder)
    filenames = subfiles(labelsTr_folder, suffix='.nii.gz', join=False)
    import matplotlib.pyplot as plt
    for n in filenames:
        img = sitk.GetArrayFromImage(sitk.ReadImage(join(imagesTr_folder, n[:-7] + '_0000.nii.gz')))
        lab = sitk.GetArrayFromImage(sitk.ReadImage(join(labelsTr_folder, n)))
        assert np.all([i == j for i, j in zip(img.shape, lab.shape)])
        z_slice = img.shape[0] // 2
        img_slice = img[z_slice]
        lab_slice = lab[z_slice]
        lab_slice[lab_slice != 0] = 1
        img_slice = img_slice - img_slice.min()
        img_slice = img_slice / img_slice.max()
        stacked = np.vstack((img_slice, lab_slice))
        print(stacked.shape)
        plt.imsave(join(output_folder, n[:-7] + '.png'), stacked, cmap='gray')

def print_unique_labels_and_their_volumes(image: str, print_only_if_vol_smaller_than: float = None):
    img = sitk.ReadImage(image)
    voxel_volume = np.prod(img.GetSpacing())
    img_npy = sitk.GetArrayFromImage(img)
    uniques = [i for i in np.unique(img_npy) if i != 0]
    volumes = {i: np.sum(img_npy == i) * voxel_volume for i in uniques}
    print('')
    print(image.split('/')[-1])
    print('uniques:', uniques)
    for k in volumes.keys():
        v = volumes[k]
        if print_only_if_vol_smaller_than is not None and v > print_only_if_vol_smaller_than:
            pass
        else:
            print('k:', k, '\tvol:', volumes[k])

out_path = "./Task037_VerSe20binary/"
data_path = "/data/userdisk0/ywye/nnUNet_raw/nnUNet_raw_data/VerSe20/"
if os.path.exists(out_path):
    shutil.rmtree(out_path)
os.makedirs(out_path, exist_ok=True)
image_path = os.path.join(out_path, "imagesTr")
label_path = os.path.join(out_path, "labelsTr")
os.makedirs(image_path, exist_ok=True)
os.makedirs(label_path, exist_ok=True)
os.makedirs(os.path.join(out_path, "imagesTs"), exist_ok=True)



all_data_name= []
sub_fork = ["dataset-01training", "dataset-02validation", "dataset-03test"]
for sub_fork_name in sub_fork:
    print(sub_fork_name)
    image_raw_path = os.path.join(data_path, sub_fork_name, "rawdata")
    label_raw_path = os.path.join(data_path, sub_fork_name, "derivatives")
    for sub_name in tqdm(os.listdir(image_raw_path)):
        if ".DS_Store" in sub_name:
            continue
        raw_data_path = os.listdir(os.path.join(image_raw_path, sub_name))
        raw_data_path = [path for path in raw_data_path if path.endswith("nii.gz")]
        assert len(raw_data_path) == 1
        shutil.copyfile(os.path.join(image_raw_path, sub_name, raw_data_path[0]), os.path.join(image_path,  raw_data_path[0].split("_")[0]+"_0000.nii.gz"))
        raw_label_path = os.listdir(os.path.join(label_raw_path, sub_name))
        raw_label_path = [path for path in raw_label_path if path.endswith("nii.gz")]
        assert len(raw_label_path) == 1
        shutil.copyfile(os.path.join(label_raw_path, sub_name, raw_label_path[0]), os.path.join(label_path, raw_data_path[0].split("_")[0] + ".nii.gz"))
        all_data_name.append(raw_data_path[0].split("_")[0])

import random
random.seed(1234)
random.shuffle(all_data_name)
print("shuffle!")
val_data_name = all_data_name[int(0.8*len(all_data_name)):]
training_data_name =  all_data_name[:int(0.8*len(all_data_name))]
print("training:", len(training_data_name), "test:", len(val_data_name))
print(training_data_name)
print(val_data_name)

json_dict = OrderedDict()
json_dict['name'] = "VerSe2020"
json_dict['description'] = "VerSe2020"
json_dict['tensorImageSize'] = "4D"
json_dict['reference'] = "see challenge website"
json_dict['licence'] = "see challenge website"
json_dict['release'] = "0.0"
json_dict['modality'] = {
    "0": "CT",
}
json_dict['labels'] = {i: str(i) for i in range(2)}

json_dict['numTraining'] = len(all_data_name)
json_dict['numTest'] = []
json_dict['training'] = [
    {'image': "./imagesTr/%s.nii.gz" % i.split("/")[-1], "label": "./labelsTr/%s.nii.gz" % i.split("/")[-1]} for i
    in
    all_data_name]
json_dict['test'] = ["./imagesTs/%s.nii.gz" % i.split("/")[-1] for i in []]

save_json(json_dict, os.path.join(out_path, "dataset.json"))

# now we reorient all those images to ras. This saves a pkl with the original affine. We need this information to
# bring our predictions into the same geometry for submission
reorient_all_images_in_folder_to_ras(image_path, 16)
reorient_all_images_in_folder_to_ras(os.path.join(out_path, "imagesTs"), 16)
reorient_all_images_in_folder_to_ras(label_path , 16)

# sanity check
check_if_all_in_good_orientation(image_path, label_path, join(out_path, 'sanitycheck'))
# looks good to me - proceed

p = Pool(6)
_ = p.starmap(print_unique_labels_and_their_volumes, zip(subfiles(label_path, suffix='.nii.gz'), [1000] * 113))

for name in tqdm(os.listdir(image_path)):
    if "nii.gz" in name:
        seg, spacing_label, origin_label, direction_label = NiiDataRead(os.path.join(label_path, name.replace("_0000", "")))
        image, spacing_img, origin_img, direction_img = NiiDataRead(os.path.join(image_path, name))


        # assert np.max(seg) == 1
        if seg.GetSize()[2] > 500:
            image  = resize_image_itk(image, newSize=[image.GetSize()[0], image.GetSize()[1], 500], resamplemethod=sitk.sitkLinear)
            seg = resize_image_itk(seg, newSize=[image.GetSize()[0], image.GetSize()[1], 500])

        print(name, image.GetSize(), spacing_img)
        assert image.GetSize() == seg.GetSize()
        # spacing_img, origin_img, direction_img = seg.GetSpacing(), seg.GetOrigin(), seg.GetDirection()
        # print(seg.GetSize(), spacing_img, origin_img, direction_img)
        seg = sitk.GetArrayFromImage(seg).astype(np.float32)
        seg[seg > 1] = 1
        assert np.max(seg) == 1
        NiiDataWrite(os.path.join(label_path, name.replace("_0000", "")), seg, spacing_img, origin_img, direction_img)
        NiiDataWrite_itk(os.path.join(image_path, name), image, spacing_img, origin_img, direction_img)

data_split_train_val = OrderedDict()
training_list = training_data_name
testing_list = val_data_name
training_list = [k.replace(".nii.gz","") for k in training_list]
testing_list = [k.replace(".nii.gz","") for k in testing_list]
with open(os.path.join(out_path, "splits_final.pkl"), "wb") as pk:
    data_split_train_val["train"] = training_list
    data_split_train_val["val"] = testing_list
    pickle.dump([data_split_train_val], pk)

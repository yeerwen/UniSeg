import torch
import nibabel as nib
import os
import numpy as np
from medpy.metric import hd95
import argparse
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool


parser = argparse.ArgumentParser()
parser.add_argument('--result_path', type=str, default='/media/new_userdisk0/SemiSL/', help='Name of Experiment')

args = parser.parse_args()

save_path = args.result_path
# gt_path = "/data/userdisk1/erwen/"
gt_path = save_path.replace("fold_0/validation_raw/", "gt_niftis/")
# gt_path = save_path.replace("all/validation_raw/", "gt_niftis/")
isHD = False
print(save_path)
txt_file = open(os.path.join(save_path, "result.txt"), "w")
task_id_dic = {"live": 0, "kidn": 1, "hepa": 2, "panc": 3, "colo": 4, "lung": 5, "sple": 6, "sub-": 7, "pros": 8, "BraT": 9, "PETC": 10}
num_class_task =  [3, 3, 3, 3, 2, 2, 2, 2, 2, 4, 2]
num_test = [27, 42, 61, 57, 26, 13, 9, 43, 25, 251, 101]
def task_index(name):
    return task_id_dic[name]

def compute_HD95(ref, pred):
    """
    ref and gt are binary integer numpy.ndarray s
    spacing is assumed to be (1, 1, 1)
    :param ref:
    :param pred:
    :return:
    """
    num_ref = np.sum(ref)
    num_pred = np.sum(pred)

    if num_ref == 0:
        if num_pred == 0:
            return 0
        else:
            return 998.
    elif num_pred == 0 and num_ref != 0:
        return 998.
    else:
        return hd95(pred, ref, (1, 1, 1))

def dice_score(preds, labels):
    preds = preds[np.newaxis, :]
    labels = labels[np.newaxis, :]
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
    predict = preds.view().reshape(preds.shape[0], -1)
    target = labels.view().reshape(labels.shape[0], -1)

    if np.sum(target) == 0 and np.sum(predict) == 0:
        return 1.0
    else:
        num = np.sum(np.multiply(predict, target), axis=1)
        den = np.sum(predict, axis=1) + np.sum(target, axis=1)

        dice = 2 * num / den

        return dice.mean()

def evaluation(root, i):
    print("Processing %s" % (i))
    i_file = os.path.join(root, i)
    i2_file = os.path.join(gt_path, i)
    predNII = nib.load(i_file)
    labelNII = nib.load(i2_file)
    pred = predNII.get_fdata()
    label = labelNII.get_fdata()
    # print(label.shape, pred.shape)
    assert label.shape == pred.shape, [label.shape, pred.shape]
    # print("label", np.min(label), np.max(label))
    # print("pred", np.min(pred), np.max(pred))
    task_id = task_index(i[:4])
    task_class = num_class_task[task_id]
    #
    count[task_id] += 1

    for sub_class in range(1, task_class):
        dice_sub_class = dice_score(pred == sub_class, label == sub_class)
        print(dice_sub_class)
        if isHD:
            HD_sub_class = compute_HD95(pred == sub_class, label == sub_class)
        else:
            HD_sub_class = 999.

        val_Dice[task_id, sub_class - 1] += dice_sub_class
        val_HD[task_id, sub_class - 1] += HD_sub_class if HD_sub_class != 999. else 0
        count_Dice[task_id, sub_class - 1] += 1
        count_HD[task_id, sub_class - 1] += 1 if HD_sub_class != 999. else 0



print("Start to evaluate...")

val_Dice = np.zeros(shape=(len(num_class_task), max(num_class_task)-1))
val_HD = np.zeros(shape=(len(num_class_task), max(num_class_task)-1))
count_Dice = np.zeros(shape=(len(num_class_task), max(num_class_task)-1))
count_HD = np.zeros(shape=(len(num_class_task), max(num_class_task)-1))
count = np.zeros(shape=(len(num_class_task),))


for root, dirs, files in os.walk(save_path):
    for i in tqdm(sorted(files)):
        if i[-6:]!='nii.gz':
            continue
        # print("Processing %s" % (i))
        evaluation(root, i)


count_Dice[count_Dice == 0] = 1
count_HD[count_HD == 0] = 1
val_Dice = val_Dice / count_Dice
val_HD = val_HD / count_HD
print(count)
for p, q in zip(count, num_test):
    if p != q and p != 0:
        exit()
print("num correct")
print("Sum results")
for t in range(len(num_class_task)):
    num_class = num_class_task[t] - 1
    log = "Sum task {}".format(t)
    for sub_class in range(num_class):
        log += " class {}, Dice: {}".format(sub_class, val_Dice[t, sub_class])

    print(log)
    txt_file.write(log+"\n")

txt_file.close()



# UniSeg-code
This is the official pytorch implementation of our MICCAI 2023 paper "[UniSeg: A Prompt-driven Universal Segmentation Model as well as A Strong Representation Learner](https://arxiv.org/pdf/2304.03493.pdf)". 
In this paper, we propose a Prompt-Driven Universal Segmentation model (UniSeg) to segment multiple organs, tumors, and vertebrae on 3D medical images with diverse modalities and domains.

<div align="center">
  <img width="100%" alt="UniSeg illustration" src="github/Overview.png">
</div>

## Requirements
CUDA 11.5<br />
Python 3.8<br /> 
Pytorch 1.11.0<br />
CuDNN 8.3.2.44

## Usage

### Installation
* Clone this repo
```
git clone https://github.com/yeerwen/UniSeg.git
cd UniSeg
```

### Data Preparation
* Download [MOTS dataset](https://github.com/jianpengz/DoDNet)
* Download [VerSe20 dataset](https://github.com/anjany/verse)
* Download [Prostate dataset](https://liuquande.github.io/SAML)
* Download [BraTS21 dataset](https://www.synapse.org/#!Synapse:syn25829067/wiki/610863)
* Download [AutoPET2022 dataset](https://autopet.grand-challenge.org)

### Pre-processing
* Step 1:
  * `cd Upstream`
  * Run `Python prepare_Kidney_Dataset.py` to normalize the name of the volumes for the Kidney dataset.
  * Run `Python Convert_MOTS_to_nnUNet_dataset.py` to pre-process the MOTS dataset.
  * Run `Python Convert_VerSe20_to_nnUNet_dataset.py` to pre-process the VerSe20 dataset and generate `splits_final.pkl`.
  * Run `Python Convert_Prostate_to_nnUNet_dataset.py` to pre-process the Prostate dataset and generate `splits_final.pkl`.
  * Run `Python Convert_BraTS21_to_nnUNet_dataset.py` to pre-process the BraTS21 dataset and generate `splits_final.pkl`.
  * Run `Python Convert_AutoPET_to_nnUNet_dataset.py` to pre-process the AutoPET2022 dataset and generate `splits_final.pkl`.

* Step 2:
  * Install nnunet by `pip install nnunet`
  * Set path, for example:
    * `export nnUNet_raw_data_base="/data/userdisk0/ywye/nnUNet_raw"`
    * `export nnUNet_preprocessed="/erwen_SSD/1T/nnUNet_preprocessed"`
    * `export RESULTS_FOLDER="/data/userdisk0/ywye/nnUNet_trained_models"`
  * Copy `Upstream/nnunet` to replace `nnunet`, which is installed by `pip install nnunet` (the address is usually 'anaconda3/envs/your envs/lib/python3.8/site-packages/nnunet').
  * Run `nnUNet_plan_and_preprocess -t 91 --verify_dataset_integrity --planner3d MOTSPlanner3D`
  * Run `nnUNet_plan_and_preprocess -t 37 --verify_dataset_integrity --planner3d VerSe20Planner3D`
  * Run `nnUNet_plan_and_preprocess -t 20 --verify_dataset_integrity --planner3d ProstatePlanner3D`
  * Run `nnUNet_plan_and_preprocess -t 21 --verify_dataset_integrity --planner3d BraTS21Planner3D`
  * Run `nnUNet_plan_and_preprocess -t 11 --verify_dataset_integrity --planner3d AutoPETPlanner3D`
  * Move `splits_final.pkl` of each dataset to the address of its pre-processed dataset. For example, '***/nnUNet_preprocessed/Task091_MOTS/splits_final.pkl'. Note that, to follow [DoDNet](https://github.com/jianpengz/DoDNet), we provide `splits_final.pkl` of the MOTS dataset in `Upstream/MOTS_data_split/splits_final.pkl`.
  * Run `Python merge_each_sub_dataet.py` to form a new dataset.
  * To make sure that we use the same data split, we provide the final data split in `Upstream/splits_final_11_tasks.pkl`


### Training 
* Move `Upstream/run_ssl.sh` and `Upstream/UniSeg_Metrics_test.py` to `"***/nnUNet_trained_models/"`
* cd `***/nnUNet_trained_models/`
* Run `sh run_ssl.sh` for training (GPU Memory Cost: ~10GB, Time Cost: ~210s each epoch).

### Pretrained weights 
* Upstream trained model is available in [UniSeg_11_Tasks](https://drive.google.com/file/d/1Ldgd5Ebc8VQrvGIpIgzUG2PTSDPUpEQJ/view?usp=sharing).

### Downstream Tasks
* Comming soon.

## To do
- [x] Dataset Links
- [x] Pre-processing Code
- [x] Upstream Code Release
- [x] Upstream Trained Model
- [ ] Downstream Code Release


## Citation
If this code is helpful for your study, please cite:

```
@article{ye2023uniseg,
  title={UniSeg: A Prompt-driven Universal Segmentation Model as well as A Strong Representation Learner},
  author={Yiwen Ye, Yutong Xie, Jianpeng Zhang, Ziyang Chen, and Yong Xia},
  booktitle={Medical Image Computing and Computer Assisted Intervention -- MICCAI 2023},
  year={2023}
}
```

## Acknowledgements
The whole framework is based on [nnUNet v1](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1).

## Contact
Yiwen Ye (ywye@mail.nwpu.edu.cn)

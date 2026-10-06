# TMo in GLAMR

## Installation instructions

Clone the repository and install the requirements.
```
conda env create -f environment.yml
```

## Data preparation

Please refer to [original GLAMR AMASS processing code](https://github.com/NVlabs/GLAMR/blob/main/preprocess/preprocess_amass.py). Additionally, we filter out treadmill sequences (sequences with `treadmill` or `normal` in `BioMotionLab_NTroje`, and `dg_07-01` in `MPI_HDM05`). 

## Run evaluation

The trained checkpoint can be found [here](https://drive.google.com/file/d/1kYabhJHLPxaqtGyYRZCMBQrmc_OJPsMO/view?usp=drive_link). 

```
python eval.py --action run --path <checkpoint path> --save <path to save the result> --dataset <dataset name>
```

## Run training code
```
python train.py --cfg traj_pred_demo --max_epochs 2000
```


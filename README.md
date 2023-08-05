# BMRC-with-DA-and-TF

#### Requirements:

```
  python==3.8.5
  torch==1.9.0+cu111
  transformers==4.8.2
```

#### Datasets:

You can download the 14-Res, 14-Lap, 15-Res, 16-Res datasets from https://github.com/xuuuluuu/SemEval-Triplet-data.
Put it into different directories (./data/original/v2) according to the version of the dataset.


#### How to run:

```
  python ./tools/Main.py --mode train # For training
  python ./tools/Main.py --mode test # For testing
```
Training different versions of datasets can modify the value of dataset_version in Main.py.
```
dataset_version = "v2/"
```


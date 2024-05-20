# 新页面

# Aspect sentiment triplet extraction based on data augmentation and task feedback

***

#### Author: Shu Liu, Tingting Lu, Kaiwen Li, Weihua Liu

The framework of the BMRC-with-DA-and-TF:

![](image/structure-eps-converted-to_00_-sZ5Z08tpu.png)

#### Requirements:

- python==3.8.5
- torch==1.9.0

#### Datasets:

You can download the 14-Res, 14-Lap, 15-Res, 16-Res datasets from [https://github.com/xuuuluuu/SemEval-Triplet-data](https://github.com/xuuuluuu/SemEval-Triplet-data "https://github.com/xuuuluuu/SemEval-Triplet-data").
Put it into different directories (./data/original/v2) according to the version of the dataset.

#### How to run:

```python
python ./tools/Main.py --mode train # For training
python ./tools/Main.py --mode test # For testing
```

Training different versions of datasets can modify the value of dataset\_version in [Main.py](http://Main.py "Main.py").

```python
dataset_version = "v2/"
```

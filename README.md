# UniRumor

## Dependencies

```python
pip install -r requirements.txt
```

## Dataset
The Raw Pheme dataset can be obtained from https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078.

The raw Weibo dataset can be downloaded from https://github.com/majingCUHK/Rumor_GAN.

The Politifact, Gossipcop dataset can be auto download bt Pytorch_Geometric.

The WeiboCOVID19, TwitterCOVID19 dataset can be download from https://drive.google.com/drive/folders/1gvuSeorLAljGZaD7gyWrUA0gyotT_rl6?usp=sharing.

The DRWeiboV3 dataset can be download from https://github.com/CcQunResearch/DRWeibo.

## Usage

You can use the following command, and the parameters are given

```python
python train.py --dataset DRWeiboV3
```

The `--dataset` argument should be one of [DRWeiboV3, Weibo, WeiboCOVID19, PHEME, Politifact, Gossipcop, TwitterCOVID19].

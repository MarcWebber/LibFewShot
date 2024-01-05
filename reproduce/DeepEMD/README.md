# Differentiable Earth Mover’s Distance for Few-Shot Learning

## Introduction

| Name:   | DeepEMD        |
| ------- | -------------- |
| Embed.: | Resnet12       |
| Type:   | Metric         |
| Venue:  | CVPR'20        |
| Codes:  | [**deep_emd**] |

Cite this work with:

```bibtex
@InProceedings{Zhang_2020_CVPR,
    author = {Zhang, Chi and Cai, Yujun and Lin, Guosheng and Shen, Chunhua},
    title = {DeepEMD: Few-Shot Image Classification With Differentiable Earth Mover's Distance and Structured Classifiers},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2020}
}

@misc{zhang2020deepemdv2,
    title={DeepEMD: Differentiable Earth Mover's Distance for Few-Shot Learning},
    author={Chi Zhang and Yujun Cai and Guosheng Lin and Chunhua Shen},
    year={2020},
    eprint={2003.06777v3},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Results and Models

**Classification**

|      | Setups   | :book: *mini*ImageNet (5,1) | :computer: *mini*ImageNet (5,1)                              | :book:*mini*ImageNet (5,5) | :computer: *mini*ImageNet (5,5)                              | :memo: Comments                     |
| ---- | -------- | --------------------------- | ------------------------------------------------------------ | -------------------------- | ------------------------------------------------------------ | ----------------------------------- |
| 1    | FCN      | -                           | 63.60± 0.36[:clipboard:](./R2D2-miniImageNet--ravi-Conv64F-5-1-Table2.yaml) | -                          | 80.20± 1.12[:clipboard:](./R2D2-miniImageNet--ravi-Conv64F-5-5-Table2.yaml) | 5-5由于训练和测试过慢，误差可能较大 |
| 2    | GRID     | -                           | 64.48± 0.39[:clipboard:](./R2D2-miniImageNet--ravi-resnet12-5-1-Table2.yaml) | -                          |                                                              |                                     |
| 3    | SAMPLING | -                           | 63.90 ± 0.38  [:clipboard:](./R2D2-miniImageNet--ravi-resnet18-5-1-Table2.yaml) | -                          |                                                              |                                     |


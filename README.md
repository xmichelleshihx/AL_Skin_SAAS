# An Active Learning Approachfor Reducing Annotation Cost in Skin Lesion Analysis
[SAAS.pdf] (https://link.springer.com/content/pdf/10.1007%2F978-3-030-32692-0_72.pdf) (MLMI@MICCAI2019)
## Introduction
Automated skin lesion analysis is very crucial in clinical practice, as skin cancer is among the most common human malignancy. Existing approaches with deep learning have achieved remarkable performance on this challenging task, however, heavily relying on large-scale labelled datasets. In this paper, we present a novel active learning framework for cost-effective skin lesion analysis. The goal is to effectively select and utilize much fewer labelled samples, while the network can still achieve state-of-the-art performance. Our sample selection criteria complementarily consider both informativeness and representativeness, derived from decoupled aspects of measuring model certainty and covering sample diversity. To make wise use of the selected samples, we further design a simple yet effective strategy to aggregate intra-class images in pixel space, as a new form of data augmentation. We validate our proposed method on data of ISIC 2017 Skin Lesion Classification Challenge for two tasks. Using only up to 50% of samples, our approach can achieve state-of-the-art performances on both tasks, which are comparable or exceeding the accuracies with full-data training, and outperform other well-known active learning methods by a large margin.
<td><img src="Screenshot from 2020-07-17 04-17-00.png" width=960 height=480></td>

## Requirements
- Python 3.6.8
- keras 
## Usage
1.  download data from [Cholec80](http://camma.u-strasbg.fr/datasets) and Resnet101 model.
2.  train with SA selection metric
    ```
    ./src/train_ud0.xSelect.sh 
    ```
3.  train with SA+AS 
    ```
    ./src/train_ud0.xSelectWithaug.sh 
    ```

# GSCon

Gradient and Structure Consistency in Multimodal Emotion Recognition

QingHongYa Shi, Mang Ye, Wenke Huang, Weijian Ruan, Bo Du, Xiaofen Zong.

## Abstract

Multimodal emotion recognition is a task that integrates text, visual, and audio data to holistically infer an individual’s emotional state. Existing research predominantly focuses on exploiting modality-specific cues for joint learning, often ignoring the differences between multiple modalities under common goal learning. Due to multimodal heterogeneity, common goal learning inadvertently introduces optimization biases and interaction noise. To address above challenges, we propose a novel approach named Gradient and Structure Consistency (GSCon). Our strategy operates at both overall and individual levels to consider balance optimization and effective interaction respectively. At the overall level, to avoid the optimization suppression of a modality on other modalities, we construct a balanced gradient direction that aligns each modality’s optimization direction, ensuring unbiased convergence. Simultaneously, at the individual level, to avoid the interaction noise caused by multimodal alignment, we align the spatial structure of samples in different modalities. The spatial structure of the samples will not differ due to modal heterogeneity, achieving effective inter-modal interaction. Extensive experiments on multimodal emotion recognition and multimodal intention understanding datasets demonstrate the effectiveness of the proposed method.

## Method Overview

![image](https://github.com/ShiQingHongYa/GSCon/blob/master/images/method.png)

The overview of gradient and structure consistency (GSCon). GSCon learns more effective common information from both the overall and individual levels. At the overall level, the gradient update direction between modalities is more consistent by the GDC. At the individual level, the spatial structure of samples between different modalities is consistent by SSC.

## Environment

Install all required libraries:

```sh
pip install -r requirements.txt
```

## Quick Start

```sh
# training
python train.py 
# evaluation
python test.py
```

## Reference

# Edge-SAN (Edge-Segment Any Nuclei)

## Introduction

In medical image analysis, accurate nuclei instance segmentation is crucial yet challenging due to variations in tissue types, staining techniques, imaging conditions, and the densely packed nature of adjacent nuclei. While the Segment Anything Model (SAM) excels in natural image settings and shows potential for medical imaging, it struggles with nuclei images due to significant differences in data distribution regarding object size, shape, and modality.

Existing medical adaptations of SAM primarily focus on organs rather than nuclei, which exhibit greater heterogeneity and clustering in large numbers. To overcome these limitations, we introduce the Cluster Edge Segment Any Nuclei (SE-SAN) approach.

## Our Contributions

1. **Unified Nuclei Dataset**: We curated a large, unified nuclei image dataset from 11 public sources to fine-tune SAM's encoder adapters and decoder. This helps bridge the data distribution gap between natural and nuclei images, enhancing the model's ability to handle diverse tissue types, staining techniques, and imaging conditions.

2. **Semi-Supervised Training**: To further enhance the model with extensive unannotated data, we implemented semi-supervised training using iterative pseudo-labeling on a dataset comprising 550K cell images. We have open-sourced the annotations of this dataset to facilitate further research and development.

3. **Novel Edge Prompt**: We proposed a novel edge prompt to improve nuclei edge delineation by identifying the touching edges of adjacent nuclei, significantly enhancing instance segmentation in densely packed clusters.

## Results

Extensive experiments validate our model's effectiveness on test images across 11 datasets and in zero-shot scenarios on MoNuSeg. Our approach is designed for easy integration into existing clinical workflows.

## Visual Representation

![Edge-SAN Visualization](https://github.com/deep-geo/NucleiSAM/assets/112611011/7a4452c0-db0c-4249-8ce4-23e7e2c78a7e)


## Datasets

### Supervised Datasets 

https://huggingface.co/datasets/DeepNuc/EdgeNuclei


## Citation

If you find this work useful for your research, please consider citing:

```bibtex
@misc{wu2023edgesan,
  author = {Xuening Wu and Yiqing Shen and Yan Wang and Qing Zhao and Yanlan Kang and Ruiqi Hu and Wenqiang Zhang},
  title = {Edge-SAN: A Nuclei Segmentation Foundation Model with Edge Prompting for Pathology Images},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/deep-geo/NucleiSAM/}}
}
```

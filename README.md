# Edge-SAN (Edge-Segment Any Nuclei)

## Introduction
Accurate nuclei segmentation is crucial for
extracting quantitative information from histology images
to support disease diagnosis and treatment decisions.
However, precise segmentation remains challenging due to
the presence of clustered nuclei, varied morphologies, and
the need to capture global spatial correlations. While stateof-the-art Transformer-based models have made progress
by employing specialized tri-decoder architectures that
separately segment nuclei, edges, and clustered edges,
their complexity and long inference times hinder integration into clinical workflows. To address this issue, we introduce MoE-NuSeg, an innovative Swin Transformer network
that incorporates domain-knowledge-driven Mixture of Experts (MoEs) to simplify the tri-decoder into a single unified
decoder. Concretely, MoE-NuSeg employs three specialized experts, each dedicated to a specific segmentation
task: nuclei identification, normal edge delineation, and
cluster edge detection. By sharing attention heads across
experts, this design efficiently mirrors the functionality of
tri-decoders while surpassing their performance and reducing the number of parameters. Furthermore, we propose a
novel two-stage training strategy to enhance performance.
In the first stage, each expert is independently trained
to specialize in their dedicated task without the gating
network. The second stage then fine-tunes the interaction
between the experts with a learnable gating network that
adaptively allocates their task-specific contributions based
on the input features. Evaluations across three datasets
spanning two modalities demonstrate that MoE-NuSeg outperforms current state-of-the-art methods, achieving an
average increase of 0.99% in Dice coefficient and 1.14%
in IoU, while reducing model parameters and FLOPs by
30.1% and 40.2%, respectively. The code is available at
https://github.com/deep-geo/MoE-NuSeg

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

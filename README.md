# Edge-SAN (Edge-Segment Any Nuclei)

In medical image analysis, accurate nuclei instance segmentation is crucial yet challenging due to variations in tissue types, staining techniques, imaging conditions, and the densely packed nature of adjacent nuclei. While the Segment Anything Model excels in natural image settings and shows potential for medical imaging, it struggles with nuclei images due to significant different data distribution in object size, shape, and modality. Exist- ing medical adaptations of SAM primarily focus on organs rather than nuclei, which exhibit greater heterogeneity and clustering in large numbers. To overcome these limitations, we introduce the Cluster Edge Segment Any Nuclei (SE- SAN) approach. Our contributions are threefold: (1) We curated a large, unified nuclei image dataset from 11 public sources to fine-tune SAM’s encoder adapters and decoder. This helps bridge the data distribution gap between natural and nuclei images, enhancing the model’s ability to han- dle diverse tissue types, staining techniques, and imaging conditions. (2) To further enhance the model with extensive unannotated data, we implemented semi-supervised train- ing using iterative pseudo-labeling on a dataset comprising 550K cell images. We have open-sourced the annotations of this dataset to facilitate further research and develop- ment. (3) We proposed a novel edge prompt to improve nuclei edge delineation by identifying the touching edges of adjacent nuclei, significantly enhancing instance segmen- tation in densely packed clusters. Extensive experiments validate our model’s effectiveness on test images across 11 datasets and in zero-shot scenarios on MoNuSeg. Our approach is designed for easy integration into existing clinical workflows. 



![Arch](https://github.com/deep-geo/NucleiSAM/assets/112611011/117bb67a-f1ab-4f0b-88cf-6a36126e9041)

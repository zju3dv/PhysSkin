# [CVPR 2026] PhysSkin: Real-Time and Generalizable Physics-Based Animation via Self-Supervised Neural Skinning


### [Project Page]() | [Arxiv]() | [Supplementary]()

> PhysSkin: Real-Time and Generalizable Physics-Based Animation via Self-Supervised Neural Skinning
> Yuanhang Lei, Tao Cheng, Xingxuan Li, Boming Zhao, Siyuan Huang, Ruizhen Hu, Peter Yichen Chen, Hujun Bao, Zhaopeng Cui†

![teaser](https://raw.githubusercontent.com/huahuo359/open_access_assets/main/PhysSkin-CVPR2026/PhysSkin_teaser.png)
Abstract: *Achieving real-time physics-based animation that generalizes across diverse 3D shapes and discretizations remains a fundamental challenge. We introduce PhysSkin, a physics-informed framework that addresses this challenge. In the spirit of Linear Blend Skinning, we learn continuous skinning fields as basis functions lifting motion subspace coordinates to full-space deformation, with subspace defined by handle transformations. 
To generate mesh-free, discretization-agnostic, and physically consistent skinning fields that generalize well across diverse 3D shapes, PhysSkin employs a new neural skinning fields autoencoder which consists of a transformer-based encoder and a cross-attention decoder.
Furthermore, we also develop a novel physics-informed self-supervised learning strategy that incorporates on-the-fly skinning-field normalization and conflict-aware gradient correction, enabling effective balancing of energy minimization, spatial smoothness, and orthogonality constraints.
PhysSkin shows outstanding performance on generalizable neural skinning and enables real-time physics-based animation.*


## Method Overview
![pipeline](https://raw.githubusercontent.com/huahuo359/open_access_assets/main/PhysSkin-CVPR2026/PhysSkin_pipeline.png)

## ToDos
🔥 Feel free to raise any requests~
- [ ] Release project page.
- [ ] Release paper.
- [ ] Release codes.

## Acknowledgement
This codebase used lots of source code from: 
1. [Simplicts](https://github.com/NVIDIAGameWorks/kaolin/tree/master/kaolin/physics/simplicits)
2. [Michelangelo](https://github.com/NeuralCarver/Michelangelo)
3. [ConFIG](https://github.com/tum-pbs/ConFIG)

We thank the authors of these projects.

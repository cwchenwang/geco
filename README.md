# GECO: Generative Image-to-3D within a SECOnd

### [Project Page](https://cwchenwang.github.io/geco) | [Paper](https://arxiv.org/abs/2405.20327)

Abstract: *3D generation has seen remarkable progress in recent years. Existing techniques, such as score distillation methods, produce notable results but require extensive per-scene optimization, impacting time efficiency. Alternatively, reconstruction-based approaches prioritize efficiency but compromise the quality due to their limited handling of uncertainty. We introduce GECO, a novel method for high-quality 3D generative modeling that operates within a second. Our approach addresses the prevalent issues of uncertainty and inefficiency in current methods through a two-stage approach. In the initial stage, we train a single-step multi-view generative model with score distillation. Then, a second-stage distillation is applied to address the challenge of view inconsistency from the multi-view prediction. This two-stage process ensures a balanced approach to 3D generation, optimizing both quality and efficiency. Our comprehensive experiments demonstrate that GECO achieves high-quality image-to-3D generation with an unprecedented level of efficiency.*

## Citation
If you consider our paper or code useful, please cite our paper:
```
@article{wang2024geco,
  title={GECO: Generative Image-to-3D within a Second},
  author={Wang, Chen and Gu, Jiatao and Long, Xiaoxiao and Liu, Yuan and Liu, Lingjie},
  journal={arXiv},
  year={2024}
}
```

## Credit
Our code is build upon from [Zero123Plus](https://github.com/SUDO-AI-3D/zero123plus) and [LGM](https://github.com/3DTopia/LGM). Thanks the authors for opensourcing.
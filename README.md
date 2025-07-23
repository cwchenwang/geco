# GECO🦎: Generative Image-to-3D within a SECOnd
**Cleaned version, please open an issue if you found any problems while running**.

### [Project Page](https://cwchenwang.github.io/geco) | [Code](https://github.com/cwchenwang/geco)

Abstract: *3D generation has seen remarkable progress in recent years. Existing techniques, such as score distillation methods, produce notable results but require extensive per-scene optimization, impacting time efficiency. Alternatively, reconstruction-based approaches prioritize efficiency but compromise the quality due to their limited handling of uncertainty. We introduce GECO, a novel method for high-quality 3D generative modeling that operates within a second. Our approach addresses the prevalent issues of uncertainty and inefficiency in current methods through a two-stage approach. In the initial stage, we train a single-step multi-view generative model with score distillation. Then, a second-stage distillation is applied to address the challenge of view inconsistency from the multi-view prediction. This two-stage process ensures a balanced approach to 3D generation, optimizing both quality and efficiency. Our comprehensive experiments demonstrate that GECO achieves high-quality image-to-3D generation with an unprecedented level of efficiency.*

## Installation
```bash
python3.10 -m venv GECO
source ./GECO/bin/activate
pip install -r requirements.txt
```

## Inference
```bash
python infer.py big --pipeline zero123plus1step --generate_mv --num_input_views 6 --bg 1.0 --cond_t 0 --init_t 950 --joint --config_path configs/instant-mesh-base.yaml --zero123plus_v 'v11' --fovy 50 --distance 2.5 --resume_pkl ./ckpts/network-snapshot-005000.pkl --resume ./ckpts/model.safetensors --output_dir ./results/test --test_path ./examples/animal_character.png
```

## Training
First stage training:
```bash
cd first-stage; bash train.sh
```

Generating data for second stage training:
```bash
python sample_data.py --sample_bs 64 --sample_data_path YOUR_DATASET --sample_start 0 --sample_end 40000 --sample_output_path SAVED_DATA_PATH
```
```bash
python sample_data_onestep.py --sample_bs 64 --sample_data_path SAVED_DATA_PATH --sample_start 0 --sample_end 40000
```

Second stage training:
```bash
accelerate launch --config_file acc_configs/gpu4.yaml main.py big --workspace logs/include-input/ --num_input_views 6 --num_views 10 --num_epochs 5 --bg 1.0 --batch_size 1 --lr 1e-6 --save_freq 1 --eval_freq 1 --stage 2 --data_path SAVED_DATA_PATH --config_path configs/instant-mesh-base.yaml --zero123plus_v 'v11' --mixed_precision 'no'
```

## Citation
If you consider our paper or code useful, please cite our paper:
```
@article{wang2024geco,
  title={GECO: Generative Image-to-3D within a Second},
  author={Wang, Chen and Gu, Jiatao and Long, Xiaoxiao and Liu, Yuan and Liu, Lingjie},
  year={2025}
}
```

## Credit
Our code is build upon from [InstantMesh](https://github.com/TencentARC/InstantMesh), [Zero123Plus](https://github.com/SUDO-AI-3D/zero123plus) and [LGM](https://github.com/3DTopia/LGM). Thanks the authors for opensourcing.
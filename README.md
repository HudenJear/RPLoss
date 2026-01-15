# RNA-Protein Loss 

This is a open source repository for RNA-Protein Loss, which aims for deep learning based RNA sequence optimization. It is still under development, thus, it may not contain all code.

## Structure

```
RNA-loss-github-togo/
├── RNA_opt/              # Main package
│   ├── test.py          # Test/inference entry point
│   ├── model/           # Model definitions
│   │   ├── arch/        # Network architectures
│   │   ├── loss/        # Loss functions (basic implementations only)
│   │   ├── metrics/     # Evaluation metrics
│   │   └── RPLoss/      # RNA-specific loss functions (placeholders)
│   └── data/            # Data loading utilities
├── options/             # Configuration YAML files
├── pretrained_weights/  # Place pre-trained model weights here
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── LICENSE             # MIT License
```

## Requirements

- Python 3.10+ (3.12 recommanded)
- PyTorch
- PyYAML
- NumPy
- To see full requirements, refer to the "requirements.txt"

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Place your pre-trained model weights in the `pretrained_weights/` directory. Download the pretrained loss weights from: [Google Drive](https://drive.google.com/file/d/1RkJ-U22sH4z2_vWHPW8EeFtkkbaOE6Nf/view?usp=drive_link)

3. Update the configuration file `options/test_rnaopt.yml`:
   - Set the path to your test data in `datasets.test.txt_path`
   - Set the path to your pre-trained model in `path.pretrain_network_g`

4. Run inference:
   ```bash
   python RNA_opt/test.py -opt options/test_rnaopt.yml
   ```

## License

See LICENSE file for details (MIT License).

## Cite us

If you find this repo useful for your work, please cite:
```
@misc{gong2025newdeeplearningbasedapproachmrna,
      title={A New Deep-learning-Based Approach For mRNA Optimization: High Fidelity, Computation Efficiency, and Multiple Optimization Factors}, 
      author={Zheng Gong and Ziyi Jiang and Weihao Gao and Deng Zhuo and Lan Ma},
      year={2025},
      eprint={2505.23862},
      archivePrefix={arXiv},
      primaryClass={q-bio.QM},
      url={https://arxiv.org/abs/2505.23862}, 
}
```

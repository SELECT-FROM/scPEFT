# scPEFT

This is the official repository for **scPEFT: Parameter-Efficient Fine-Tuning Enhances Adaptation of Single Cell Large Language Model.**

[![Preprint](https://img.shields.io/badge/preprint-available-brightgreen)](https://www.biorxiv.org/content/10.1101/2024.01.27.577455v1) &nbsp;

## Installation

scPEFT works with Python >= 3.7.13. Please make sure you have the correct version of Python installed pre-installation.

scPEFT is available on PyPI. To install scPEFT, run the following command:

```bash
pip install scpeft
```

For developing, run the following command:

```
git clone https://github.com/SELECT-FROM/scPEFT
cd scPEFT
```



## Get Started

1. Download the upstream model  [scGPT model checkpoint](https://github.com/bowang-lab/scGPT/blob/main/README.md#pretrained-scgpt-model-zoo) and place it at e.g., `work_dir/scPEFT/save`. We recommend using the [whole-human](https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y?usp=sharing) model for most applications by default, which pretrained on 33 million normal human cells..

2. The tutorials of scPEFT for downstream tasks in  [tutorial_peft](https://github.com/SELECT-FROM/scPEFT/tree/main/tutorial_peft).  Here are the links to the downstream tasks and tutorials mentioned in our article

   | Downstream task          | Link                                                         |
   | :----------------------- | :----------------------------------------------------------- |
   | cell type identification | [Tutorial_Identification.ipynb](https://github.com/SELECT-FROM/scPEFT/blob/main/tutorial_peft/Tutorial_Identification.ipynb) |
   | batch correction         | [Tutorial_BatchCorrection.ipynb](https://github.com/SELECT-FROM/scPEFT/blob/main/tutorial_peft/Tutorial_BatchCorrection.ipynb) |
   | perturbation             | [Tutorial_Perturbation.ipynb](https://github.com/SELECT-FROM/scPEFT/blob/main/tutorial_peft/Tutorial_Perturbation.ipynb) |
   | case control             | [Tutorial_CaseControl.ipynb](https://github.com/SELECT-FROM/scPEFT/blob/main/tutorial_peft/Tutorial_Perturbation.ipynb) |

   

## Contributing

We greatly welcome contributions to scPEFT. Please submit a pull request if you have any ideas or bug fixes. We also welcome any issues you encounter while using scPEFT.

## Built With

We sincerely thank the authors of following open-source projects:

- [scGPT](https://github.com/bowang-lab/scGPT)
- [scanpy](https://github.com/scverse/scanpy)
- [scvi-tools](https://github.com/scverse/scvi-tools)
- [scib](https://github.com/theislab/scib)
- [datasets](https://github.com/huggingface/datasets)
- [transformers](https://github.com/huggingface/transformers)

## Citing scPEFT

```bibtex
@article {He2024.01.27.577455,
	author = {Fei He and Ruixin Fei and Mingyue Gao and Li Su and Xinyu Zhang and Dong Xu},
	title = {Parameter-Efficient Fine-Tuning Enhances Adaptation of Single Cell Large Language Model for Cell Type Identification},
	year = {2024},
	doi = {10.1101/2024.01.27.577455},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2024/01/30/2024.01.27.577455},
	journal = {bioRxiv}
}
```

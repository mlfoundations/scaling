# Language models scale reliably with over-training and on downstream tasks

[[Project page](https://mlfoundations.github.io/scaling/)] [[Arxiv](https://arxiv.org/abs/2403.08540)]

This repository contains code for our paper [Language models scale reliably with over-training and on downstream tasks](https://arxiv.org/abs/2403.08540).

To understand the scaling behavior of over-trained models (i.e., models trained past compute-optimal), we train 104 models between 11M and 6.9B parameters on three datasets (C4, RedPajama, and RefinedWeb), with various token budgets. We conduct 8 next-token loss evaluations to construct scaling laws that relate training compute, model parameters, and the amount of over-training to next-token loss. To better align scaling laws with downstream accuracy/error metrics, we additionally evaluate all models on 46 downstream tasks. We propose a power law that connects a model's perplexity to its downstream average top-1 error.

We hope you will have fun playing with our scaling testbed!

If you have any questions, please contact [Samir](https://sagadre.github.io) at `sy [at] cs [dot] columbia [dot] edu` or open an issue if you feel others can also benefit from the discussion!

**Table of Contents**

- [Environment](#environment)
- [Examining our testbed](#examining-our-testbed)
- [Downloading models and inference](#downloading-models-and-inference)
- [Training logs](#training-logs)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

# Environment

Create the conda environment:
```sh
conda env create -f environment.yml
```
Activate the environment:
```sh
conda activate scaling
```

# Examining our testbed
We release our loss evaluations in `exp_data/model/*` and downstream accuracy/error evaluations in `exp_data/evals/*`.
To make interacting with these evaluations simpler, we provide `scaling_explorer.ipynb`, which contains examples. Specifically, we include cells to:
* Load data for all 104 models, with 8 loss evaluations and 46 downstream evaluations per model.
* Fit loss and downstream scaling laws following the default configurations from Table 2 in our paper.
* Code to reproduce Figure 1 demonstrating our key results.

We hope that you will consider extending this notebook!

# Downloading models and inference
We release model weights here: https://huggingface.co/mlfoundations/scaling/tree/main

WARNING: These are base models and not aligned with post-training. They are provided as is and intended as research artifacts only.

To automatically download a model and run inference, for instance, for the 411M parameter model trained on RefinedWeb for approximately 32x compute-optimal (i.e., with a token multiplier of 640),
```
python generate.py --model-json exp_data/models/rw_original-d=1024_l=24_h=8-32.0.json --input-text <your prompt in quotes here>
```

# Training logs
For downloading training logs and loading (number of token token, training loss) pairs per run see `training_explorer.ipynb`. Because of the decentrialized nature of our runs, there are unfortunately a few training logs that are not present in our wandb.

# Acknowledgements

SYG is supported by an NSF Graduate Research Fellowship, GS by the Onassis Foundation - Scholarship ID: F ZS 056-1/2022-2023, and MN by the Federal Ministry of Education and Research of Germany under grant no. 01IS22094B WEST-AI. We thank Stability AI and Toyota Research Institute (TRI) for access to compute resources. This research has been supported by NSF Grants AF 1901292, CNS 2148141, Tripods CCF 1934932, IFML CCF 2019844, and research gifts by Western Digital, Amazon, WNCG IAP, UT Austin Machine Learning Lab (MLL), Cisco, and the Stanly P. Finch Centennial Professorship in Engineering. We also thank Kushal Arora, Alper Canberk, Mia Chiquier, Sachit Menon, Chuer Pan, Purva Tendulkar, and Mandi Zhao for valuable feedback.

# Citation

If you find this code or our paper useful, consider citing:

```bibtex
@article{gadre2024LanguageMS,
    author = {Gadre, Samir Yitzhak and Smyrnis, Georgios and Shankar, Vaishaal and Gururangan, Suchin and Wortsman, Mitchell and Shao, Rulin and Mercat, Jean and Fang, Alex and Li, Jeffrey and Keh, Sedrick and Xin, Rui and Nezhurina, Marianna and Vasiljevic, Igor and Jitsev, Jenia and Dimakis, Alexandros G. and Ilharco, Gabriel and Song, Shuran and Kollar, Thomas and Carmon, Yair and Dave, Achal and Heckel, Reinhard and Muennighoff, Niklas and Schmidt, Ludwig},
    title= {Language models scale reliably with over-training and on downstream tasks},
    year = {2024},
    journal = {arXiv preprint},
    note = {\url{https://arxiv.org/abs/2403.08540}}
}
```

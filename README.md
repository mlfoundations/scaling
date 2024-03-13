<h1>Language models scale reliably with over-training and on downstream tasks</h1>
<div style="text-align: left;">

Scaling laws are useful guides for developing language models, but there are still gaps between current scaling studies and how language models are ultimately trained and evaluated. For instance, scaling is usually studied in the compute-optimal training regime (i.e., "Chinchilla optimal" regime); however, in practice, models are often over-trained to reduce inference costs. Moreover, scaling laws mostly predict loss on next-token prediction, but ultimately models are compared based on downstream task performance. In this paper, we address both shortcomings. To do so, we create a testbed of 104 models with 0.011B to 6.9B parameters trained with various numbers of tokens on three data distributions. First, we investigate scaling in the over-trained regime. We fit scaling laws that extrapolate in both the number of model parameters and the ratio of training tokens to parameters. This enables us to predict the validation loss of a 1.4B parameter, 900B token run (i.e., 32x over-trained) and a 6.9B parameter, 138B token run---each from experiments that take 300x less compute. Second, we relate the perplexity of a language model to its downstream task performance via a power law. We use this law to predict top-1 error averaged over downstream tasks for the two aforementioned models using experiments that take 20x less compute. 

</div>
</div>

<br>

This repository contains code for messin' around with our scaling laws testbed.

If you have any questions, please contact [Samir](https://sagadre.github.io) at `sy [at] cs [dot] columbia [dot] edu` or open an issue if you feel others can also benefit from the discussion!

**Table of Contents**

- [Environment](#environment)
- [Examining our testbed](#examining-our-testbed)
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

We hope that you will consider extending this notebook and find something new that we missed!

# Acknowledgements

SYG is supported by an NSF Graduate Research Fellowship, GS by the Onassis Foundation - Scholarship ID: F ZS 056-1/2022-2023, and MN by the Federal Ministry of Education and Research of Germany under grant no. 01IS22094B WEST-AI. We thank Stability AI and Toyota Research Institute (TRI) for access to compute resources. This research has been supported by NSF Grants AF 1901292, CNS 2148141, Tripods CCF 1934932, IFML CCF 2019844, and research gifts by Western Digital, Amazon, WNCG IAP, UT Austin Machine Learning Lab (MLL), Cisco, and the Stanly P. Finch Centennial Professorship in Engineering. We also thank Kushal Arora, Alper Canberk, Mia Chiquier, Sachit Menon, Chuer Pan, Purva Tendulkar, and Mandi Zhao for valuable feedback.

# Citation

If you find this code or our paper useful, consider citing:

```bibtex
@article{gadre2024LanguageMS,
    author = {Gadre, Samir Yitzhak and Smyrnis, Georgios and Shankar, Vaishaal and Gururangan, Suchin and Wortsman, Mitchell and Shao, Rulin and Mercat, Jean and Fang, Alex and Li, Jeffrey and Keh, Sedrick and Xin, Rui and Nezhurina, Marianna and Vasiljevic, Igor and Jitsev, Jenia and Dimakis, Alexandros G. and Ilharco, Gabriel and Song, Shuran and Kollar, Thomas and Carmon, Yair and Dave, Achal and Heckel, Reinhard and Muennighoff, Niklas and Schmidt, Ludwig},
    title= {Language models scale reliably with over-training and on downstream tasks},
    year = {2024},
    journal = {Preprint}
}
```
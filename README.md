# advbench

This repository contains the code needed to reproduce the results of the following papers:

* "Adversarial Robustness with Semi-Infinite Constrained Learning" (NeurIPS 2021) by Alexander Robey, Luiz F.O. Chamon, George J. Pappas, Hamed Hassani, and Alejandro Ribeiro.  
* "Probabilistically Robust Learning: Balancing Average and Worst-case Performance" (ICML 2022) by Alexander Robey, Luiz F. O. Chamon, George J. Pappas, and Hamed Hassani

If you find this repository useful in your research, please consider citing:

```
@article{robey2021adversarial,
  title={Adversarial robustness with semi-infinite constrained learning},
  author={Robey, Alexander and Chamon, Luiz and Pappas, George J and Hassani, Hamed and Ribeiro, Alejandro},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={6198--6215},
  year={2021}
}
@inproceedings{robey2022probabilistically,
  title={Probabilistically Robust Learning: Balancing Average and Worst-case Performance},
  author={Robey, Alexander and Chamon, Luiz and Pappas, George J and Hassani, Hamed},
  booktitle={International Conference on Machine Learning},
  pages={18667--18686},
  year={2022},
  organization={PMLR}
}
```

---

### Overview

This repository contains code for reproducing our results, including implementations of each of the baseline algorithms used in our paper.  At present, we support the following baseline algorithms:

* Empirical risk minimization (ERM, [Vapnik, 1998](https://www.wiley.com/en-fr/Statistical+Learning+Theory-p-9780471030034))
* Projected gradient ascent (PGD, [Madry et al., 2017](https://arxiv.org/abs/1706.06083))
* Fast gradient sign method (FGSM, [Goodfellow et al., 2014](https://arxiv.org/abs/1412.6572))
* Clean logit pairing (CLP, [Kannan et al., 2018](https://arxiv.org/abs/1803.06373))
* Adversarial logit pairing (ALP, [Kannan et al., 2018](https://arxiv.org/abs/1803.06373))
* Theoretically principled trade-off between robustness and accuracy (TRADES, [Zhang et al., 2019](https://arxiv.org/abs/1901.08573))
* Misclassification-aware adversarial training (MART, [Wang et al., 2020](https://openreview.net/forum?id=rklOg6EFwS))

We also support several versions of our own algorithm.

* Dual Adversarial Learning with Gaussian prior (Gaussian_DALE)
* Dual Adversarial Learning with Laplacian prior (Laplacian_DALE)
* Dual Adversarial Learning with KL-divergence loss (KL_DALE)

---

### Repository structure

The structure of this repository is based on the (excellent) [domainbed](https://github.com/facebookresearch/DomainBed) repository.  All of the runnable scripts are located in the `advbench.scripts/` and `advbench.plotting` directories.

---

### Quick start

Train a model:

```
python -m advbench.scripts.train --dataset CIFAR10 --algorithm KL_DALE_PD --output_dir train-output --evaluators Clean PGD
```

Tally the results:

```
python -m advbench.scripts.collect_results --depth 0 --input_dir train-output
```
# Robust Lottery Tickets for Pre-trained Language Models

Code for our ACL 2022 paper :

Rui Zheng , Rong Bao, Yuhao Zhou et al., *[Robust Lottery Tickets for Pre-trained Language Models
](https://aclanthology.org/2022.acl-long.157.pdf)*


## Dependencies
- Python 3.8
- [PyTorch](https://github.com/pytorch/pytorch) 1.7.1
- [TextAttack](https://github.com/QData/TextAttack) 0.3.4

## Installation Instructions

It is suggested to use a python 3.8 environment to run the experiment. If you're using conda, create a new environment with:

```
conda create --name robust_ticket python=3.8
```

Activate the newly created environment with:

```
conda activate robust_ticket
```

And install the required packages by:

```
pip install -r requirements.txt
```

This should install both Tensorflow and tf_metrics.

## Usage

**Stage 1: Fine-tuning BERT on downstream tasks**
Fine-tuneing BERT on downstream tasks using officical default settings.
```
sh stage1-finetune.sh
```

**Stage 2: Searching robust ticktes using adversarial training** 
```
sh stage2-search-ticket.sh
```

**Stage 3: Drawing and re-training robust ticktes**
```
sh stage3-draw-retrain_ticket.sh
```

## Citation
```
@inproceedings{zheng-etal-2022-robust,
    title = "Robust Lottery Tickets for Pre-trained Language Models",
    author = "Zheng, Rui  and
      Rong, Bao  and
      Zhou, Yuhao  and
      Liang, Di  and
      Wang, Sirui  and
      Wu, Wei  and
      Gui, Tao  and
      Zhang, Qi  and
      Huang, Xuanjing",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.157",
    doi = "10.18653/v1/2022.acl-long.157",
    pages = "2211--2224",
    abstract = "Recent works on Lottery Ticket Hypothesis have shown that pre-trained language models (PLMs) contain smaller matching subnetworks(winning tickets) which are capable of reaching accuracy comparable to the original models. However, these tickets are proved to be notrobust to adversarial examples, and even worse than their PLM counterparts. To address this problem, we propose a novel method based on learning binary weight masks to identify robust tickets hidden in the original PLMs. Since the loss is not differentiable for the binary mask, we assign the hard concrete distribution to the masks and encourage their sparsity using a smoothing approximation of L0 regularization.Furthermore, we design an adversarial loss objective to guide the search for robust tickets and ensure that the tickets perform well bothin accuracy and robustness. Experimental results show the significant improvement of the proposed method over previous work on adversarial robustness evaluation.",
}
```

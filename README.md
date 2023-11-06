# UCE
Uncertainty-Aware Encoder (Findings of EMNLP 2021)

This repo holds the code for our Uncertainty-Aware Encoder, UCE, described in our Findings of EMNLP 2021 paper: "[An Uncertainty-Aware Encoder for Aspect Detection
](https://aclanthology.org/2021.findings-emnlp.69.pdf)" 

## Installation

```
pip install -r requirements.txt
```

## Citations

```
@inproceedings{nguyen-etal-2021-uncertainty-aware,
    title = "An Uncertainty-Aware Encoder for Aspect Detection",
    author = "Nguyen, Thi-Nhung  and
      Nguyen, Kiem-Hieu  and
      Song, Young-In  and
      Cao, Tuan-Dung",
    editor = "Moens, Marie-Francine  and
      Huang, Xuanjing  and
      Specia, Lucia  and
      Yih, Scott Wen-tau",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-emnlp.69",
    doi = "10.18653/v1/2021.findings-emnlp.69",
    pages = "797--806",
    abstract = "Aspect detection is a fundamental task in opinion mining. Previous works use seed words either as priors of topic models, as anchors to guide the learning of aspects, or as features of aspect classifiers. This paper presents a novel weakly-supervised method to exploit seed words for aspect detection based on an encoder architecture. The encoder maps segments and aspects into a low-dimensional embedding space. The goal is approximating similarity between segments and aspects in the embedding space and their ground-truth similarity generated from seed words. An objective function is proposed to capture the uncertainty of ground-truth similarity. Our method outperforms previous works on several benchmarks in various domains.",
}
```

# GraphBTM

This repository contains the code for ourl EMNLP 2018 paper "GraphBTM: Graph Enhanced Autoencoded Variational Inference for Biterm Topic Model", you can find [here](http://www.aclweb.org/anthology/D18-1495).

Some code are based on the pytorch implementation of AVITM: 
https://github.com/hyqneuron/pytorch-avitm
The Topic Coherence Evaluation is from: 
https://github.com/jhlau/topic_interpretability
Thanks for sharing code!

# Requirements
  - python 3.6
  - pytorch 0.4
  - numpy
  - python 2.7 for topic coherence evaluation

# How to use
```sh
$ python pytorch_run.py --start
```
It may take some time to generate the biterms (it's too large to upload the pickle, so I upload the original files).
It will generate the top 10 words in each topic after each epoch in 'topic_interpretability/data/topics_20news.txt', and you can use the code in the topic_interpretability folder:
```sh
$ ./run-oc.sh
```

If you find the code helpful, please kindly cite the paper:
~~~~~
> @InProceedings{D18-1495,
  author = 	"Zhu, Qile and Feng, Zheng and Li, Xiaolin",
  title = 	"GraphBTM: Graph Enhanced Autoencoded Variational Inference for Biterm Topic Model",
  booktitle = 	"Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
  year = 	"2018",
  publisher = 	"Association for Computational Linguistics",
  pages = 	"4663--4672",
  location = 	"Brussels, Belgium",
  url = 	"http://aclweb.org/anthology/D18-1495"
}

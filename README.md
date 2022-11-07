# Somun: entity-centric summarization incorporating pre-trained language models

Text summarization resolves the issue of capturing essential information from a large volume of text data. Existing methods either depend on the end-to-end models or hand-crafted preprocessing steps. In this study, we propose an entity-centric summarization method which extracts named entities and produces a small graph with a dependency parser. To extract entities, we employ well-known pre-trained language models. After generating the graph, we perform the summarization by ranking entities using the harmonic centrality algorithm. Experiments illustrate that we outperform the state-of-the-art unsupervised learning baselines by improving the performance more than 10% for ROUGE-1 and more than 50% for ROUGE-2 scores. Moreover, we achieve comparable results to recent end-to-end models.

## If you want to use Somun in your studies, please cite the following paper:

```
@article{inan2021somun,
  title={Somun: entity-centric summarization incorporating pre-trained language models},
  author={Inan, Emrah},
  journal={Neural Computing and Applications},
  volume={33},
  number={10},
  pages={5301--5311},
  year={2021},
  publisher={Springer}
}
```

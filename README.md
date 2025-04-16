# cs768-2025-assignment

## Assignment: CS 768

### Notes:
1. Evaluation files will be uploaded on 20th April. Work on your solutions until then.
2. git clone this repository. It contains the dataset `dataset_papers.tar.gz`
3. Revisit this repository on 21st April for updates.
4. Work in teams according to your projects.

### Setup
In this assignment, we will work on a real world application, a problem that is naturally modelled as a graph.
You are given a dataset of research papers from NeurIPS and ICML from several past years some even two decades old. [Download here](https://drive.google.com/file/d/1J73io_KqCoPEAlH3teLWGoZ78yk5n7ll/view?usp=sharing). The dataset contains ~6500 folders, each containing data of a research paper. Specifically the folder contains title.txt, abstract.txt and bibliography in either .bbl, .bib or both formats.

A sample paper folder:
```bash
./<paper_name>/
├── abstract.txt
├── something.bbl
├── optionally.bib
└── title.txt
```

**Note:** This dataset has been manually curated by the TAs and does not exist elsewhere (we got blocked by arXiv in the process). Consequently, it may have some noisy entries.

### Task 1: Build a citation graph
First, use the references in the bibliography to generate a citation graph. The graph can either be directed or undirected. After building the graph, report the following:
1. The number of edges in the graph.
2. Number of isolated nodes in the graph.
3. The average degree (in-degree, out-degree). Plot a histogram of the degrees of nodes.
4. Diameter of the graph.

### Task 2: Machine Learning
Feel free to use any approach of your choice for this part.

In this part, you have to train a model to look at a new unseen paper and return a ranked list of papers (from our dataset) which this unseen paper may have cited. This is essentially a link prediction task. Moreover, the evaluation of your predictions will be based on recall@K metric (Value of K will not be released). Specifically,
1. Update the `evaluation.py` file that takes as argument ‘paper_folder_path’.
2. You are supposed to fill this file with your code and print your top-K predicted papers to the console in the specific format.
`evaluation.py` will be called by a caller file `run_evaluation.py`. This file will feed papers to `evaluation.py` and get top-K predictions. If one of the top-K papers is indeed in the citations list, then you get a score.


### Submission Requirements:
1. Github of your codebase.
2. Report of your approach for both task 1 and task 2.

### Grading
- 5 marks for your model performance
- 5 marks for report
- 5 marks for your codebase
- 5 marks for uniqueness, novelty and soundness of your approach.
Total marks = 20.

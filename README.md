# Drug-Drug Interaction Predictor with Graph Neural Networks

Predicts likely interactions between drugs from FDA adverse-event reports. Classical models come first as baselines, then a graph neural network learns from the structure of which drugs appear together.

*Independent portfolio project, FDA FAERS data*

## Result

**78.1% AUC** with the graph neural network, measured against classical baselines built first.

## What I did

- **Data:** cleaned and standardized FDA FAERS adverse-event data, including mapping brand names to generic drug names so the same drug isn't counted twice.
- **Baselines first:** trained classical models before the graph model, so any gain could be measured honestly.
- **Graph model:** built a GNN that represents drugs and their co-occurrence as a graph, to capture interaction structure.
- **Analysis:** error analysis and ablation tests to find which features and graph relationships actually improved predictions.
- **Documentation:** wrote up modeling assumptions, trade-offs and negative results, so the work can be reproduced.

## Tech stack

Python, PyTorch, PyTorch Geometric, pandas, scikit-learn, Jupyter

## How to run

<!-- Faryal: add the notebook name, where to download the FAERS data, and setup steps here. -->

## Author

**Faryal Bahawi** | [Portfolio](https://faryal-bahawi.netlify.app) | [LinkedIn](https://www.linkedin.com/in/faryal-bahawi-28027b205/)

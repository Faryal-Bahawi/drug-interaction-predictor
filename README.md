# Drug-Drug Interaction Predictor

Taking two medications together can cause reactions that neither causes alone, and a lot of those interactions only show up after a drug is already on the market. The FDA collects millions of adverse-event reports in its FAERS database, which made me wonder: if two drugs keep appearing together in reports of the same reactions, can a model learn to flag that pair?

This project is my attempt at that. I treat drugs as points in a network, connect the ones with known interactions, and train a graph neural network to predict which other pairs are likely to interact. The final model reaches **78.1% AUC**.

There's also a small app. Type in two medications, for example warfarin and fluconazole, and it gives you an interaction score and the number of adverse reactions the two drugs share in FDA reports.

> This is an educational project, not medical advice. Always check with a doctor or pharmacist before combining medications.

## How I approached it

**Cleaning the data came first.** FAERS reports use brand names and generic names interchangeably, so the same drug can appear under several names. I standardized everything to generic names so a drug isn't counted twice.

**Then I built simple baselines.** Before touching the graph model I trained classical models, so I'd know whether the GNN was actually adding anything.

**Then the graph model.** Each drug is a node described by three features: its average proportional reporting ratio (PRR), how many reactions it typically shares with other drugs, and how many known interactions it has. A two-layer graph convolutional network learns an embedding for every drug, and the model scores a pair of drugs by how closely their embeddings line up.

**Finally, I checked what really helped.** Error analysis and ablation tests showed which features and graph connections improved predictions and which didn't. I've kept the negative results in the report too.

## What's in this repository

| File | What it is |
|---|---|
| `data_exploration.ipynb` | Data cleaning, exploration, baselines and model training |
| `app.py` | The Streamlit app for checking two drugs |
| `best_drug_gnn.pt` | The trained model weights |
| `drug_to_idx.json`, `drug_names.json` | Lookup tables the app uses to find each drug |
| `Drug_Interaction_Predictor_Report.docx` | The full project report |
| `Drug_Interaction_Predictor_Slides.pptx` | Presentation slides |

## Running the app

```bash
git clone https://github.com/Faryal-Bahawi/drug-interaction-predictor.git
cd drug-interaction-predictor
pip install -r requirements.txt
streamlit run app.py
```

The app also reads a local SQLite database, `faers.db`, with the processed FAERS tables (`model_data` and `drug_reaction_signals`). It isn't included in this repository, so you'll need to build it from the FAERS data first.

## Built with

Python, PyTorch, PyTorch Geometric, NetworkX, pandas, scikit-learn, SQLite and Streamlit.

---

Made by **Faryal Bahawi**. More of my work is on my [portfolio](https://faryal-bahawi.netlify.app), and I'm on [LinkedIn](https://www.linkedin.com/in/faryal-bahawi-28027b205/).

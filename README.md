# Drug-Drug Interaction Predictor

A machine learning system that predicts dangerous drug-drug interactions using Graph Neural Networks trained on 5 million real FDA adverse event reports.

## Live Demo
Try the app: [Coming soon — Hugging Face deployment]

## Results
| Model | Metric | Score |
|---|---|---|
| Logistic Regression | Accuracy | 58% |
| Random Forest | Accuracy | 60% |
| Graph Neural Network | Test AUC | 78.1% |

The GNN outperformed classical ML by 30% by learning from drug network structure rather than individual pair statistics.

## Real Predictions
| Drug Pair | Score | Clinical Note |
|---|---|---|
| Cyclosporine + Fluconazole | 99.7% | Raises cyclosporine toxicity — kidney damage risk |
| Warfarin + Fluconazole | 99.0% | Significantly increases bleeding risk |
| Haloperidol + Clozapine | 99.8% | Detected via graph structure |
| Warfarin + Ibuprofen | 72.1% | Lower but real risk |

## Dataset
- Source: FDA FAERS 2026 Q1 (downloaded directly from FDA)
- 5+ million adverse event reports across 7 tables
- Drug interaction labels from DrugBank (ChCh-Miner dataset)
- 29,557 unique drug names standardized via RxNorm API

## Methodology
1. Data cleaning — regex, RxNorm standardization, noise removal
2. PRR analysis — same statistical method used by the FDA
3. Classical ML baseline — Logistic Regression, Random Forest
4. Graph Neural Network — drugs as nodes, interactions as edges
5. Proper train/validation/test split (70/10/20)
6. Streamlit web app deployment

## Tech Stack
Python • SQLite • Pandas • NetworkX • PyTorch Geometric • Streamlit • RxNorm API • FDA FAERS • DrugBank

## Files
- `01_data_exploration.ipynb` — full data pipeline and model training
- `app.py` — Streamlit web application
- `best_drug_gnn.pt` — trained GNN model (78.1% Test AUC)
- `drug_to_idx.json` — drug name to index mapping
- `drug_names.json` — list of 544 drugs in the graph
- `Drug_Interaction_Predictor_Report_v2.docx` — full technical report
- `Drug_Interaction_Predictor_Slides_v2.pptx` — presentation slides

## Disclaimer
This tool is for educational purposes only. Not a substitute for professional medical advice.

## Author
Faryal Bahawi | Data Science M.S. Candidate | University of New Haven

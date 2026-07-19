import streamlit as st
import torch
import json
import pandas as pd
import sqlite3
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

st.set_page_config(
    page_title="Drug Interaction Predictor",
    page_icon="💊",
    layout="centered"
)

st.markdown("""
<style>
.stButton>button {
    background-color: #1A2B3C;
    color: white;
    border-radius: 8px;
    padding: 10px 24px;
    font-size: 16px;
    width: 100%;
    border: none;
}
.stButton>button:hover {
    background-color: #00C2A8;
}
</style>
""", unsafe_allow_html=True)

class DrugGNN(torch.nn.Module):
    def __init__(self):
        super(DrugGNN, self).__init__()
        self.conv1 = GCNConv(3, 64)
        self.conv2 = GCNConv(64, 32)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_index):
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

@st.cache_resource
def load_everything():
    with open('drug_to_idx.json', 'r') as f:
        drug_to_idx = json.load(f)
    with open('drug_names.json', 'r') as f:
        drug_names = json.load(f)

    conn = sqlite3.connect('faers.db')
    model_data = pd.read_sql_query("SELECT * FROM model_data", conn)
    drug_signals = pd.read_sql_query("SELECT * FROM drug_reaction_signals", conn)
    conn.close()

    # Rebuild node features
    import networkx as nx
    interactions = model_data[model_data['label'] == 1]
    G = nx.Graph()
    for _, row in interactions.iterrows():
        G.add_edge(row['drug1_name'], row['drug2_name'])

    node_features = []
    for drug in drug_names:
        drug_data = model_data[
            (model_data['drug1_name'] == drug) |
            (model_data['drug2_name'] == drug)
        ]
        avg_prr = drug_data['avg_prr_drug1'].mean() if len(drug_data) > 0 else 0
        avg_reactions = drug_data['shared_reactions'].mean() if len(drug_data) > 0 else 0
        degree = G.degree(drug) if drug in G else 0
        node_features.append([avg_prr, avg_reactions, degree])

    x = torch.tensor(node_features, dtype=torch.float)

    edge_index = []
    for drug1, drug2 in G.edges():
        idx1 = drug_to_idx[drug1]
        idx2 = drug_to_idx[drug2]
        edge_index.append([idx1, idx2])
        edge_index.append([idx2, idx1])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    model = DrugGNN()
    model.load_state_dict(torch.load('best_drug_gnn.pt', map_location=torch.device('cpu')))
    model.eval()

    with torch.no_grad():
        z = model.encode(x, edge_index)

    return drug_to_idx, drug_names, model_data, drug_signals, z

drug_to_idx, drug_names, model_data, drug_signals, z = load_everything()

# App UI
st.markdown("# 💊 Drug-Drug Interaction Predictor")
st.markdown("##### Powered by Graph Neural Networks | Trained on FDA FAERS data (5M+ reports)")
st.markdown("---")

st.markdown("### Check two medications for interactions")

col1, col2 = st.columns(2)
with col1:
    drug1 = st.text_input("First medication", placeholder="e.g. warfarin")
with col2:
    drug2 = st.text_input("Second medication", placeholder="e.g. fluconazole")

check = st.button("Check Interaction")

if check:
    drug1 = drug1.lower().strip()
    drug2 = drug2.lower().strip()

    if not drug1 or not drug2:
        st.warning("Please enter both drug names.")
    elif drug1 == drug2:
        st.warning("Please enter two different drug names.")
    elif drug1 not in drug_to_idx or drug2 not in drug_to_idx:
        st.error("One or both drugs not found. Try generic names like 'warfarin', 'fluconazole', 'cyclosporine', 'haloperidol'.")
    else:
        idx1 = drug_to_idx[drug1]
        idx2 = drug_to_idx[drug2]
        edge = torch.tensor([[idx1], [idx2]], dtype=torch.long)

        with torch.no_grad():
            score = torch.sigmoid(
                (z[edge[0]] * z[edge[1]]).sum(dim=-1)
            ).item()

        # Get shared reactions
        pair = model_data[
            ((model_data['drug1_name'] == drug1) & (model_data['drug2_name'] == drug2)) |
            ((model_data['drug1_name'] == drug2) & (model_data['drug2_name'] == drug1))
        ]
        shared = int(pair.iloc[0]['shared_reactions']) if len(pair) > 0 else 0

        st.markdown("---")
        st.markdown(f"### Results: **{drug1.title()}** + **{drug2.title()}**")

        col_score, col_reactions = st.columns(2)
        with col_score:
            st.metric("Interaction Score", f"{score:.1%}")
        with col_reactions:
            st.metric("Shared Reactions", shared)

        if score >= 0.5:
            st.markdown(f"""
            <div style='background-color:#fff3cd;padding:20px;border-radius:10px;border-left:5px solid #ff9800;margin-top:10px;'>
            <h3>⚠️ Potential Interaction Detected</h3>
            <p>The model predicts a <b>{score:.1%}</b> probability of interaction between <b>{drug1.title()}</b> and <b>{drug2.title()}</b>.</p>
            <p>These drugs share <b>{shared}</b> common adverse reactions in FDA reports.</p>
            <p><b>Please consult your doctor or pharmacist before taking these together.</b></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background-color:#d4edda;padding:20px;border-radius:10px;border-left:5px solid #28a745;margin-top:10px;'>
            <h3>✅ Low Interaction Risk</h3>
            <p>The model predicts a <b>{score:.1%}</b> probability of interaction between <b>{drug1.title()}</b> and <b>{drug2.title()}</b>.</p>
            <p>No significant shared adverse reactions were found in FDA reports.</p>
            <p><b>Always consult a healthcare professional before combining medications.</b></p>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<small>
⚠️ <b>Disclaimer:</b> This tool is for educational purposes only and is not a substitute for professional medical advice.
The model was trained on FDA FAERS 2026 Q1 data and achieved 78.1% AUC on unseen drug pairs.
</small>
""", unsafe_allow_html=True)
import pandas as pd
import gradio as gr
from agent import TumorClassifierAgent

# Load small TCGA subset
data = pd.read_csv("data/mini_tcga.csv")
gene_columns = [c for c in data.columns if c not in ["Sample_ID", "Tumor_Type"]]

# Initialize agent
agent = TumorClassifierAgent(genes=gene_columns)

def run_agent(sample_id):
    sample_row = data[data["Sample_ID"] == sample_id]
    features = sample_row[gene_columns].iloc[0]
    
    pred = agent.classify_sample(features)
    quest_text = agent.gamify_prediction(pred, sample_id)
    key_genes = agent.extract_key_genes(quest_text)
    
    return f"Tumor Type Prediction: {pred}\n\nGamified Quest:\n{quest_text}\n\nKey Genes Highlighted: {key_genes}"

# Gradio UI
sample_ids = data["Sample_ID"].tolist()
iface = gr.Interface(fn=run_agent, inputs=gr.Dropdown(sample_ids, label="Select Sample"), outputs="text", title="Gamified Tumor Explorer AI")
iface.launch()

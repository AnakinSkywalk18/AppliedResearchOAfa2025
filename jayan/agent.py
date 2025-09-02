import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

class TumorClassifierAgent:
    def __init__(self, model_name="distilbert-base-uncased", genes=None):
        # Load pretrained text model for fun gamification text
        self.text_model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.text_model_name)
        self.text_model = AutoModelForSequenceClassification.from_pretrained(
            self.text_model_name, num_labels=2
        )
        self.ner_pipeline = pipeline("ner")
        self.genes = genes if genes else []

    def classify_sample(self, sample_features):
        """
        Fake classifier for demo: sum features, threshold for label
        Replace with proper ML model if desired
        """
        score = sample_features.sum()
        return "BRCA" if score % 2 > 1 else "LUAD"

    def gamify_prediction(self, tumor_type, sample_id):
        """
        Generates a fun description using the tumor type
        """
        return f"Sample {sample_id} is predicted as {tumor_type}. Quest: Explore critical genes and unlock the secrets of the tumor realm!"

    def extract_key_genes(self, text):
        """
        Dummy NER on gene names
        """
        ner_results = self.ner_pipeline(text)
        # Filter for genes if they match known list
        key_genes = [r["word"] for r in ner_results if r["word"] in self.genes]
        return key_genes

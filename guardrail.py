import torch
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict
import nltk
from classifiers import LLMClassifier, BERTaClassifier
from config import ModelConfig
from transformer_lens import HookedTransformer
import os
import glob
import re

class MoralGuardrail:
    def __init__(self, target_model, llm_classifier, berta_classifier):
        self.target_model = target_model
        #self.target_tokenizer = target_tokenizer
        self.llm_classifier = llm_classifier
        self.berta_classifier = berta_classifier
        self.steering_vec = None
        self.hook_name = None
        # Initialize NLTK
        nltk.download('punkt')
        nltk.download('punkt_tab')
        
        # Create results directory
        Path(ModelConfig.RESULTS_PATH).mkdir(exist_ok=True)

    def generate_response(self, prompt: str) -> str:
        """Generate response from target model"""
        #inputs = self.target_tokenizer(prompt, return_tensors="pt").to(self.target_model.device)
        
        outputs = self.target_model.generate(
            prompt,
            max_new_tokens=ModelConfig.MAX_LENGTH,
        )
        
        return outputs.replace(prompt, "").strip()

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        return nltk.sent_tokenize(text) #split by period
        # parts = re.split(r'[.?!,:;]+', text)
        # parts = [p.strip() for p in parts if p.strip()]
        # return parts

    def analyze_prompt(self, prompt: str) -> List[float]:

        sentences = self.split_into_sentences(prompt)
        prompt_trans = []

        for sentence in sentences:
            t_mft_scores, t_i = self.berta_classifier.analyze_suffix(sentence)
            prompt_trans.append({
                "mft_scores": t_mft_scores,
                "salience": t_i
            })
        prompt_score = [r["salience"] for r in prompt_trans]
        #normalized = [(x + 1) / 2 for x in prompt_score]
            
        return prompt_score

    def analyze_response(self, response: str) -> Tuple[List[Dict[str, List[float]]], List[Dict[str, List[float]]]]:
        """
        Analyze response and return both MFT scores and salience scores
        Returns:
            Tuple containing:
            - List of dicts with norm MFT scores and salience
            - List of dicts with transgression MFT scores and salience
        """
        sentences = self.split_into_sentences(response)
        norm_results = []
        transgression_results = []
        
        prefix = ""
        for sentence in sentences:
            # Analyze prefix
            if prefix == "":
                n_mft_scores = ModelConfig.DEFAULT_MFT
                n_i = 0.0
            else:
                n_mft_scores, n_i = self.llm_classifier.analyze_prefix(prefix)
            norm_results.append({
                "mft_scores": n_mft_scores,
                "salience": n_i
            })
            
            # Analyze current sentence
            t_mft_scores, t_i = self.berta_classifier.analyze_suffix(sentence)
            transgression_results.append({
                "mft_scores": t_mft_scores,
                "salience": t_i
            })
            
            prefix = prefix + " " + sentence if prefix else sentence
            
        return norm_results, transgression_results

    def calculate_hypo_degree(self, 
                            norm_results: List[Dict[str, List[float]]], 
                            transgression_results: List[Dict[str, List[float]]]) -> List[float]:
        """Calculate hypo degree using salience scores"""
        if not norm_results or not transgression_results:
            return []
        
        norm_salience_list = [r["salience"] for r in norm_results]
        trans_salience_list = [r["salience"] for r in transgression_results]
        
        # Ensure that both salience lists have the same length.
        if len(norm_salience_list) != len(trans_salience_list):
            raise ValueError("The salience lists in norm_results and transgression_results must have the same length.")
        
        # Compute the hypo degree for each MFT dimension.
        # hypo_degrees = [norm - trans for norm, trans in zip(norm_salience_list, trans_salience_list)]
        hypo_degrees = [
            abs(norm - trans) + (abs((norm + trans)/2) if (norm < 0 and trans < 0) else 0)
            for norm, trans in zip(norm_salience_list, trans_salience_list)
        ]

        return hypo_degrees

    def load_steering_vector(self, directory: str, model_shortname: str, pattern: str = None):

        if pattern is None:
            pattern = f"{model_shortname}_layer*_steering_vector.pt"
        search_path = os.path.join(directory, pattern)
        files = glob.glob(search_path)
        if not files:
            raise FileNotFoundError(f"No file matching pattern '{pattern}' found in directory '{directory}'")
        file_path = files[0]
        self.steering_vec = torch.load(file_path)
        print(f"Loaded steering vector of shape: {self.steering_vec.shape}")

        # Use regex to extract the layer id from the filename (e.g. "layer12")
        basename = os.path.basename(file_path)
        match = re.search(r"layer(\d+)", basename)
        if not match:
            raise ValueError(f"Could not extract layer id from filename: {basename}")
        layer_id = int(match.group(1))
        self.hook_name = f"blocks.{layer_id}.hook_resid_post"
        print(f"Extracted layer id: {layer_id}. Hook will be registered at: {self.hook_name}")

        return self.steering_vec, layer_id

    def generate_steered_response(self, prompt: str, coeff: float, max_new_tokens: int = ModelConfig.MAX_LENGTH, do_sample: bool = False):

        if self.steering_vec is None or self.hook_name is None:
            raise ValueError("Steering vector not loaded. Call load_steering_vector() first.")

        def steering_hook(activation, hook):
            # Expand steering_vec from shape [hidden_dim] to [1, 1, hidden_dim]
            steering_expanded = self.steering_vec.unsqueeze(0).unsqueeze(0)
            return activation + coeff * steering_expanded

        self.target_model.add_hook(self.hook_name, steering_hook)
        output = self.target_model.generate(prompt, max_new_tokens=max_new_tokens, do_sample=do_sample)
        # Remove hooks to restore original behavior.
        self.target_model.reset_hooks()
        return output.replace(prompt, "").strip()


    def save_results(self, ind:int, model_name, results: List[Dict]):
        """Save analysis results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if ind == 1:
            output_file = Path(ModelConfig.HYPO_RESULTS_PATH) / f"detailed/{model_name}_analysis_results_{timestamp}.json"
        if ind == 2:
            output_file = Path(ModelConfig.ALIGN_RESULTS_PATH) / f"{model_name}_results_{timestamp}.json"
        if ind == 3:
            output_file = Path(ModelConfig.JAILRBREAK_RESULTS_PATH) / f"{model_name}_results_{timestamp}.json"
        else:
            output_file = Path(ModelConfig.RESULTS_PATH) / f"{model_name}_analysis_results_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        print(f"Prompts Results saved to {output_file}") 

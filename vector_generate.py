import json
import numpy as np
import pandas as pd
import torch
import gc
from pathlib import Path
from transformer_lens import HookedTransformer

class SteeringVectorSaver:
    def __init__(self, model_name: str, model_dir: str, llama_json_path: str, mfv_json_path: str):
        # Load the model
        self.model = HookedTransformer.from_pretrained(model_dir)
        
        # Load evaluation scores from llama3.1.json
        with open(llama_json_path, 'r') as f:
            self.llama_data = json.load(f)
        
        # Load texts and labels from MFV_val.json
        with open(mfv_json_path, 'r') as f:
            self.mfv_data = json.load(f)

        self.key = "all"

        
        # Select best layer
        #self.layer_id = self.model.cfg.n_layers - 1 #last layer
        self.layer_id = self.select_best_layer()
        self.hook_name = f"blocks.{self.layer_id}.hook_resid_post"
        
        # Load texts and labels from the "all" key
        self.texts, self.labels = self.load_texts_labels()
        
        # Placeholder for steering vector
        self.steering_vec = None

    def select_best_layer(self) -> int:
        """Select the layer with the highest score from the 'all' key in llama3.1.json."""
        scores = self.llama_data["all"]
        best_layer = int(np.argmax(scores))
        print(f"Selected layer {best_layer} with score {scores[best_layer]:.4f}")
        return best_layer

    def load_texts_labels(self):
        """Extract texts and labels from MFV_val.json under the 'all' key."""
        data = self.mfv_data[self.key]
        texts = [item["text"] for item in data]
        labels = [item["label"] for item in data]
        labels = np.array(labels)
        return texts, labels

    def compute_steering_vector(self):

        all_activations = []
        for text in self.texts:
            _, cache = self.model.run_with_cache(text)
            act = cache[self.hook_name].mean(dim=1).squeeze(0)  # [hidden_dim]
            all_activations.append(act)
        
        acts_tensor = torch.stack(all_activations)
        label0_acts = acts_tensor[self.labels == 0]
        label1_acts = acts_tensor[self.labels == 1]
        mean_label0 = label0_acts.mean(dim=0)
        mean_label1 = label1_acts.mean(dim=0)
        
        steering_vec = mean_label0 - mean_label1
        
        # Normalize
        self.steering_vec = steering_vec / steering_vec.norm()
        print(f"Computed steering vector shape: {self.steering_vec.shape}")
        print(f"Steering vector norm: {self.steering_vec.norm():.2f}")
        return self.steering_vec

    def save_vector(self, save_path: str):
        """Save the computed steering vector to a file."""
        if self.steering_vec is None:
            raise ValueError("Steering vector not computed yet. Call compute_steering_vector() first.")
        filepath = Path(save_path) / f"{model_name}_layer{self.layer_id}_steering_vector.pt"
        torch.save(self.steering_vec, filepath)
        print(f"Steering vector saved to {filepath}")
        # Release memory after done with the model
        del self.model
        gc.collect()    # Prompt garbage collection  

        if torch.cuda.is_available():
            torch.cuda.empty_cache()  


# Example usage for computing and saving the steering vector:
if __name__ == "__main__":
    models = ["llama2-7"]
    model_dirs = ["meta-llama/Llama-2-7b-chat-hf"]
    results_dir = "./results/lens_results"
    save_path = "./results/directions"
    mfv_json_path = "./MFV/MFV_val.json"
    #moralchoice_path="./MFV/moralchoice.csv"
    for model_name,model_dir in zip(models,model_dirs):
        lens_json_path = Path(results_dir) / f"{model_name}_lens_results.json"
        saver = SteeringVectorSaver(model_name, model_dir, lens_json_path, mfv_json_path)
        saver.compute_steering_vector()
        saver.save_vector(save_path)

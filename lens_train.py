import torch
import json
import seaborn as sns
import numpy as np
import pandas as pd
from huggingface_hub import login
import matplotlib.pyplot as plt
from datetime import datetime
from transformer_lens import HookedTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pathlib import Path
import gc

def extract_activation_for_layer(text: str, model, layer_id: int):
    # Run the text through the model and capture activations using TransformerLens.
    _, cache = model.run_with_cache(text)
    # Extract activations from the specified layer.
    activations = cache[f"blocks.{layer_id}.hook_resid_post"]
    # Average over the sequence dimension (assuming batch size 1).
    act_mean = activations.mean(dim=1)
    # Return a numpy array of shape [hidden_dim]
    return act_mean.squeeze(0).detach().cpu().numpy()

def convert_keys_to_str(d):
    if isinstance(d, dict):
        return {str(k): convert_keys_to_str(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [convert_keys_to_str(item) for item in d]
    else:
        return d


hf_token = " " #llama token
login(token=hf_token)
model_names = ["meta-llama/Llama-2-7b-chat-hf"]
models = ["llama2-7"]

df = pd.read_csv("./MFV/moralchoice.csv")
df_800 = df.sample(n=862, random_state=42)
ethics_dataset = df_800.to_dict(orient='records')
with open("./MFV/MFV_val.json", "r") as f:
    validation_sets = json.load(f)
dimensions = list(validation_sets.keys())
print("MFV sets loaded for dimensions:", dimensions)

results_dir = Path("./results/lens_results")
results_dir.mkdir(parents=True, exist_ok=True)

for model_name,model_ind in zip(model_names,models):
    file_path = Path(results_dir) / f"{model_ind}_lens_results.json"
    model = HookedTransformer.from_pretrained(model_name)
    print(f"--------------------------------------------------")
    print(f"Lens training for {model_name} start...")
    print(f"--------------------------------------------------")

    # -------------------------------
    # Step 2: Evaluate layer-wise accuracy on each validation set.
    # -------------------------------
    num_layers = model.cfg.n_layers

    # This dictionary will store results as: { dimension: [accuracy_layer0, accuracy_layer1, ..., accuracy_layerN-1] }
    results = {}

    # For each validation set (each dimension):
    for dim in dimensions:
        start_tmp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\nEvaluating validation set for dimension: {dim}. Starting at {start_tmp}")
        val_set = validation_sets[dim]
        
        # Separate texts and labels.
        train_texts = [ex["input"] for ex in ethics_dataset]
        train_labels = np.array([ex["label"] for ex in ethics_dataset])    
        val_texts = [ex["text"] for ex in val_set]       
        val_labels = np.array([ex["label"] for ex in val_set])        
        
        layer_accs = []
        
        # For each layer, extract activations for all examples and evaluate a probe.
        for layer in range(num_layers):
            # Extract activation vectors for all texts in the current validation set at this layer.
            X_ethics_en = np.stack([extract_activation_for_layer(text1, model, layer) for text1 in train_texts])
            X_mfv_en = np.stack([extract_activation_for_layer(text2, model, layer) for text2 in val_texts])
            
            # form the train and test set
            # X_train, X_test, y_train, y_test = train_test_split(X_val_en, val_labels, test_size=0.6, random_state=42)
            indices = np.arange(len(X_mfv_en))
            np.random.shuffle(indices)

            split_size = 8
            train_indices = indices[:split_size]
            test_indices = indices[split_size:]

            mfv_X2_train = X_mfv_en[train_indices]
            mfv_y2_train = np.array(val_labels)[train_indices] 
            mfv_X2_test = X_mfv_en[test_indices]
            mfv_y2_test = np.array(val_labels)[test_indices]

            X_train = np.concatenate([X_ethics_en, mfv_X2_train], axis=0)
            y_train = np.concatenate([train_labels, mfv_y2_train], axis=0)
            X_test = mfv_X2_test
            y_test = mfv_y2_test

            # Train a logistic regression classifier (the probe) on the training split.
            clf = LogisticRegression(max_iter=1000, random_state=42)
            clf.fit(X_train, y_train)
            # clf = LogisticRegression(max_iter=1000, random_state=42)
            # clf.fit(X_ethics_en, train_labels)        
            
            # Evaluate on the MFV test split.
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            layer_accs.append(acc)
            tmp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"  Layer {layer}: Accuracy = {acc:.3f} at {tmp}")

            # Update the partial results for the current dimension
            results[dim] = layer_accs
            # Save updated results to file after each layer iteration
            with open(file_path, "w") as f:
                json.dump(convert_keys_to_str(results), f, indent=2)
            print(f"Saved partial results at layer {layer}.")
        
        #results[dim] = layer_accs


    end_tmp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"Probe training for {model_ind} end at {end_tmp}.")
    print(f"Probe results saved to '{file_path}'")


    # plot heat map
    fig_path = Path(results_dir) / f"{model_ind}_lens_results.png"
    plot_dimensions = ["care", "fairness", "loyalty", "authority", "purity", "all"]
    #num_layers = model.cfg.n_layers
    data = np.array([results[dim] for dim in plot_dimensions])
    plt.figure(figsize=(12, 4))

    # Plot the heatmap
    ax = sns.heatmap(
        data,
        annot=None,         # show accuracy values in each cell
        fmt=".2f",          # format with 3 decimal places
        cmap="YlGn",        # color map
        linewidths=1,       # create grid lines
        linecolor="white",  # grid lines color
        cbar=True,          # show color bar
        cbar_kws={'pad': 0.01},
        vmin=0, vmax=1      # if your accuracies range between 0 and 1
    )


    ax.set_xticklabels([str(i) for i in range(num_layers)], rotation=45, ha="right")
    ax.set_yticklabels(dimensions, rotation=45)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=14)

    # Retrieve the colorbar object from the heatmap
    cbar = ax.collections[0].colorbar

    # Set the colorbar label with a specific font size
    cbar.set_label('Accuracy', fontsize=12, labelpad=12)
    sns.despine(left=True, bottom=True)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=800)
    plt.show()

    # Release memory after done with the model
    del model
    gc.collect()           # Prompt garbage collection

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

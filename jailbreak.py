from huggingface_hub import login
from model_loader import ModelLoader
from guardrail import MoralGuardrail
from classifiers import LLMClassifier, BERTaClassifier
from config import ModelConfig
from pathlib import Path
from datetime import datetime
import pandas as pd
import random
import json
import torch
import gc

def load_jailbreak_prompts(prompt_path):
    """Load jailbreak prompts from file"""
    file_path = prompt_path
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def append_unique_augs(augs_this_run, path):
    path = Path(path)
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                existing = json.load(f)
            if not isinstance(existing, list):
                existing = []
        except Exception:
            existing = []
    else:
        existing = []
    existing_prompts = {item.get("prompt") for item in existing if isinstance(item, dict)}

    seen_this_run = set()
    for aug in augs_this_run:
        p = aug.get("prompt")
        if p and p not in existing_prompts and p not in seen_this_run:
            existing.append(aug)
            existing_prompts.add(p)
            seen_this_run.add(p)

    with path.open("w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

def main():
    # Initialize model loader

    hf_token = ModelConfig.HF_API_KEY
    login(token=hf_token)
    model_dirs = [
        "meta-llama/Llama-2-13b-chat-hf"
    ]
    model_names = ["llama2"]

    direction_dir = ModelConfig.STEERING_VECTORS_PATH
    theta = ModelConfig.HARM_THRESHOLD
    beta = ModelConfig.STEER_DEGREE

    jb_method = ModelConfig.JB_METHOD

    for model_name, model_dir in zip(model_names, model_dirs):
        for jb in jb_method:
            augs_this_run = []

            loader = ModelLoader(model_dir)
            
            # Load models
            target_model = loader.load_target_model()
            api_client, (berta_model, berta_tokenizer) = loader.load_classifier_models()
            
            # Initialize classifiers and finder
            llm_classifier = LLMClassifier(api_client)
            berta_classifier = BERTaClassifier(berta_model, berta_tokenizer)

            # Initialize guardrail
            guardrail = MoralGuardrail(
                target_model,
                llm_classifier,
                berta_classifier
            )

            prompt_path = Path(ModelConfig.JAILBREAK_PROMPTS_PATH) / f"{model_name}_{jb}.json"
            prompts = load_jailbreak_prompts(prompt_path)
            timestamps = datetime.now().strftime("%Y%m%d_%H%M%S")
            print(f"Start jailbreak analyze for {model_name} on method {jb} at {timestamps}")
            results = []
            args = {
                "model": model_name,
                "jailbreak method": jb
            }
            results.append(args)
            
            for i, prompt in enumerate(prompts, 1):
                is_harm = False
                timestamp0 = datetime.now().strftime("%Y%m%d_%H%M%S")
                print(f"\nProcessing prompt {i}/{len(prompts)} at {timestamp0}")

                prompt_score = guardrail.analyze_prompt(prompt)
                # Generate response
                response = guardrail.generate_response(prompt)
                hypo_degree = []
                steered_response = ""
                
                if all(x > 0.4 for x in prompt_score):
                    is_harm = False
                    steered_response = response
                if sum(x < 0 for x in prompt_score) >= len(prompt_score) / 2:
                    is_harm = True
                    guardrail.load_steering_vector(direction_dir, model_name)
                    steered_response = guardrail.generate_steered_response(prompt, coeff=beta)
                else:
                    # Analyze response
                    norm_results, transgression_results = guardrail.analyze_response(response)
                    hypo_degree = guardrail.calculate_hypo_degree(norm_results, transgression_results)
                    print(f"Hypo degree for prompt {i} : {hypo_degree}")

                    if any(x > theta for x in hypo_degree) or sum(x > theta*0.75 for x in hypo_degree) >= 3:
                        is_harm = True
                        guardrail.load_steering_vector(direction_dir, model_name)
                        steered_response = guardrail.generate_steered_response(prompt, coeff=beta)
                    
                # Store results
                result = {
                    "prompt num": i,
                    "prompt": prompt,
                    "response": response,
                    "prompt score": prompt_score,
                    "is_harmful": is_harm,
                    "steered response": steered_response,
                    "norm_salience": [r["salience"] for r in norm_results],
                    "transgression_salience": [r["salience"] for r in transgression_results],
                    "hypo_degree": hypo_degree
                }
                results.append(result)

                if all(x > 0 for x in prompt_score) and is_harm:
                    aug = {
                        "prompt": prompt,
                        "prompt score": prompt_score,
                        "response": response,
                        "steered response": steered_response
                    }
                    augs_this_run = [].append(aug)


            guardrail.save_results(3, model_name, results)
            if augs_this_run:
                append_unique_augs(augs_this_run, ModelConfig.AUG_PATH)


    # guardrail.save_results(results)

if __name__ == "__main__":
    main() 
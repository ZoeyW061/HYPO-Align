import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Pipeline
from openai import OpenAI
import re
from typing import Dict, List, Tuple
from config import ModelConfig

class LLMClassifier:
    def __init__(self, client):
        # Initialize API client
        self.client = client
        
    def parse_scores(self, response: str) -> Dict[str, float]:
        """Parse the LLM response to extract MFT scores"""
        parse_scores = {}
        pattern_template = r"{}:\s*([-+]?\d+(?:\.\d+)?)"
        for dimension in ModelConfig.MFT_DIMENSIONS:
            pattern = pattern_template.format(dimension)
            match = re.search(pattern, response)
            if match:
                parse_scores[dimension] = float(match.group(1))
            else:
                parse_scores[dimension] = 0.0
        return parse_scores

    def calculate_norm_salience(self, scores: Dict[str, float]) -> float:
        """Calculate Norm Salience (N_i) from MFT scores"""
        moderate_scores = ModelConfig.MODERATE_MFT
        total = sum(moderate_scores.values())
        normalized_w = {dim: score / total for dim, score in moderate_scores.items()}
        norm_salience = sum(normalized_w[dim] * scores[dim] for dim in moderate_scores)
        return norm_salience/3.0

    def analyze_prefix(self, prefix: str) -> Tuple[Dict[str, float], float]:
        """
        Analyze prefix text and return MFT scores and Norm Salience score using DeepSeek API
        Each analysis uses a fresh context window
        Returns:
        Tuple[Dict[str, float], float]: (MFT scores dictionary, Norm Salience score)
        """
        try:
            system_prompt = ModelConfig.MORAL_SYS_PROMPT
            user_prompt = ModelConfig.MORAL_ANALYSIS_PROMPT.format(text=prefix)
            combined_prompt = f"{system_prompt}\n\n{user_prompt}" 
            response = self.client.chat.completions.create(
                model=ModelConfig.DEEPSEEK_MODEL_NAME,
                #model=ModelConfig.OPENAI_MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                    # {"role": "user", "content": combined_prompt} # for openai reasoning model
                ],
                temperature=ModelConfig.CLASSIFIER_TEMPERATURE,
                max_tokens=ModelConfig.CLASSIFIER_MAX_LENGTH,
                stop=None,  # Ensure no continuation from previous contexts
                # max_completion_tokens = ModelConfig.CLASSIFIER_MAX_LENGTH, # for openai reasoning model
                presence_penalty=0,  # Reduce influence of previous tokens
                frequency_penalty=0   # Reduce influence of token frequency
            )
            #response.raise_for_status()

            #check status
            #print(f"Status: {response.status_code}")

            # Extract the response text
            response_text = response.choices[0].message.content
            #print(f"Deepseek response is:{response_text}")

            # Parse scores and calculate Norm Salience
            scores_dict = self.parse_scores(response_text)
            #print(f"The MFT score dictionary is: {scores_dict} \n")
            
            # Convert dictionary to ordered list of scores
            mft_scores = [scores_dict[dim] for dim in ModelConfig.MFT_DIMENSIONS]
            norm_salience = self.calculate_norm_salience(scores_dict)

            return  scores_dict, norm_salience
            
        except Exception as e:
            print(f"Error in API call: {e}\n at Prompt Prefix: {prefix}")
            return {dim: 0.0 for dim in ModelConfig.MFT_DIMENSIONS}, 0.0

class BERTaClassifier:
    def __init__(self, berta_model, berta_tokenizer):
        self.model = berta_model
        self.tokenizer = berta_tokenizer
        self.device = torch.device(ModelConfig.DEVICE)

    def calculate_transgression_salience(self, scores: Dict[str, float]) -> float:
        """Calculate Transgression Salience (T_i) from classification scores"""
        index_key = max(scores, key=lambda k: abs(scores[k]))
        return scores[index_key]

    def analyze_suffix(self, suffix: str) -> Tuple[Dict[str, float], float]:
        """
        Analyze single sentence and return both MFT scores and Transgression Salience score
        Returns:
            Tuple[Dict[str, float], float]: (MFT scores dictionary, Transgression Salience score)
        """
        probs = []
        mft_scores = []
        inputs = ["Alice said: " + suffix] * len(ModelConfig.MFT_DIMENSIONS)
        labels = ModelConfig.CLASSIFIER_MFT_LABELS
        features = self.tokenizer(inputs, labels, padding=True, truncation=True, return_tensors="pt")
        features = {key: value.to(self.device) for key, value in features.items()}

        # get the logtis and probs for each value mapping ['contradiction', 'entailment', 'neutral']
        self.model.eval()
        with torch.no_grad():
            scores = self.model(**features).logits
        for score in scores:
            prob = F.softmax(score, dim=-1)
            probs.append(prob)
            p_contra = prob[0].item()
            p_entail = prob[1].item()
            p_neutral = prob[2].item()
            raw_diff = p_entail - p_contra  # in [-1, 1]
            confidence = 1 - p_neutral
            final_score = raw_diff * confidence
            mft_scores.append(final_score)
        
        #convert to mft dictionary
        score_dict = {
            dim: score
            for dim, score in zip(ModelConfig.MFT_DIMENSIONS, mft_scores)
        }
        transgression_salience = self.calculate_transgression_salience(score_dict)
        
        return score_dict, transgression_salience


            

    # def analyze_suffix(self, suffix: str) -> Tuple[List[float], float]:
    #     """
    #     Analyze single sentence and return both MFT scores and Transgression Salience score
    #     Returns:
    #         Tuple[List[float], float]: (MFT scores list, Transgression Salience score)
    #     """
    #     inputs = self.tokenizer(suffix, return_tensors="pt")
    #     outputs = self.model(**inputs)
    #     scores = torch.softmax(outputs.logits, dim=1)[0]
        
    #     # Convert to list of scores in the same order as MFT_DIMENSIONS
    #     mft_scores = [score.item() for score in scores]
        
    #     # Convert to dictionary for salience calculation
    #     score_dict = {
    #         dim: score 
    #         for dim, score in zip(ModelConfig.MFT_DIMENSIONS, mft_scores)
    #     }
        
    #     transgression_salience = self.calculate_transgression_salience(score_dict)
        
    #     return mft_scores, transgression_salience 

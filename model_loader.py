import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from transformer_lens import HookedTransformer
from openai import OpenAI
from config import ModelConfig

class ModelLoader:
    def __init__(self, model_dir):
        self.device = torch.device(ModelConfig.DEVICE)
        self.model_dir = model_dir

    def load_target_model(self):
        """Load the target model"""
        print("Loading model...")
        model = HookedTransformer.from_pretrained(
            self.model_dir,
        ).to(self.device)
        model.eval()
        # model = AutoModelForCausalLM.from_pretrained(
        #     self.model_dir,
        #     torch_dtype=torch.float16,
        #     device_map="auto"
        # )
        # tokenizer = AutoTokenizer.from_pretrained(
        #     self.model_dir,
        #     use_fast=True
        # )
        return model

    def load_classifier_models(self):
        """Load both classifier models"""
        print("Loading classifier models...")
        # Load DeepSeek
        client = OpenAI(
            api_key=ModelConfig.DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com/v1",
            #api_key = ModelConfig.OPENAI_API_KEY,
            timeout=ModelConfig.API_TIMEOUT, 
            max_retries=ModelConfig.API_MAX_RETRIES
        )

        # Load BERTa
        berta_model = AutoModelForSequenceClassification.from_pretrained(
            ModelConfig.DEBERTA_PATH
            #num_labels=len(ModelConfig.MFT_DIMENSIONS)
        ).to(self.device)
        berta_tokenizer = AutoTokenizer.from_pretrained(
            ModelConfig.DEBERTA_PATH
        )

        return client, (berta_model, berta_tokenizer)
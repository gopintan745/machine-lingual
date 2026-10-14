from transformers import AutoModelForSeq2SeqLM
from configs.config import config


def load_model():
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model.base, cache_dir=config.model.cached_dir)
    return model
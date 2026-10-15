from transformers import AutoModelForSeq2SeqLM
from configs.config import Config


def load_model(config: Config):
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model.base, cache_dir=config.model.cached_dir)
    model.add_adapter(config.lora, adapter_name="lora_adapters")
    return model
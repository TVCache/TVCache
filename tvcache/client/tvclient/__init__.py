"""TVCache - A caching library to accelerate tool calls in RL finetuning."""
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
__version__ = "0.1.0"

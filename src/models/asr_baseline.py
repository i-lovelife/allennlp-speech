from pathlib import Path
from allennlp.models.model import Model

@Model.register("asr_baseline")
class AsrBaseline(Model):
    """
    """
    def __init__(self):
        

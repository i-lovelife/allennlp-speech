from typing import Optional, Dict
from allennlp.data import Vocabulary
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, Seq2VecEncoder
from allennlp.training.metrics import BooleanAccuracy

import numpy
from overrides import overrides
import torch
import torch.nn as nn
import torch.nn.functional as F


@Model.register("asr_baseline")
class AsrBaseline(Model):
    """
    """
    def __init__(self, 
                 vocab: Vocabulary,
                 encoder: Seq2SeqEncoder,
                 classifier: Seq2VecEncoder,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(AsrBaseline, self).__init__(vocab, regularizer)
        
        num_class = vocab.get_vocab_size()
        self.encoder = encoder
        self.classifier = classifier
        self.acc = BooleanAccuracy()
        initializer(self)

    @overrides
    def forward(self,
                features: torch.Tensor,
                lable: torch.IntTensor = None) -> Dict[str, torch.Tensor]:
        """
        Parameters:
            features: extracted audio feature (batch, T, feature)
            label: correspond label (batch, vocab_size)
        """
        features.transpose_(-2, -1)# (batch, feature, T)
        hidden = self.encoder(features)# (batch, new_feature, T)
        out_logits = self.classifier(hidden)# (batch, vocab_size)
        predicted = torch.argmax(out_logits, dim=1)# (batch,)
        self.acc(predicted, lable)
        output_dict = {'out_logits': F.log_softmax(out_logits)}
        loss = F.cross_entropy(out_logits, lable)
        output_dict['loss'] = loss
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'acc':self.acc.get_metric(reset)}

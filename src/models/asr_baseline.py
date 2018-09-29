from typing import Optional, Dict, List, Any
from allennlp.data import Vocabulary
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.models.model import Model
from allennlp.nn.util import sort_batch_by_length

from overrides import overrides
import torch
import torch.nn as nn
import torch.nn.functional as F
import external.model_speech as external_model

@Model.register("asr_baseline")
class AsrBaseline(Model):
    """
    Asr baseline
    """
    def __init__(self,
                 vocab: Vocabulary,
                 rnn_type: nn.Module = nn.LSTM,
                 feature_size: int = 513,
                 rnn_hidden_size: int = 768,
                 nb_layers: int = 2,
                 bidirectional=True,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(AsrBaseline, self).__init__(vocab, regularizer)
        num_class = vocab.get_vocab_size(namespace='character')
        self.speech_model = external_model.DeepSpeech(rnn_type,
                                                      num_class,
                                                      feature_size,
                                                      rnn_hidden_size,
                                                      nb_layers,
                                                      bidirectional)
        initializer(self)

    def forward(self,
                feature: torch.Tensor,
                feature_length: torch.Tensor,
                txt_label: Dict[str, torch.Tensor] = None,
                txt_length: torch.Tensor = None,
                meta_data: List[Dict[str, Any]] = None,
                **args: Any) -> Dict[str, torch.Tensor]:
        """
        Parameters:
            feature: (batch, T, feature)
            feature_length: (batch)
            txt_label: {
                        "character": (batch, max_label_length)
                    }
            txt_length: (batch)
        """
        if txt_label is not None:
            txt_label = txt_label['character'].view(-1)
        sorted_feature, sorted_feature_length, restore_idx, _ = sort_batch_by_length(feature, feature_length)
        sorted_feature = sorted_feature.transpose(-2, -1).unsqueeze(1)# (batch, 1, feature, T)
        logits, output_lengths = self.speech_model(sorted_feature, sorted_feature_length)
        logits = logits.index_select(0, restore_idx)# (batch, T, num_class)
        output_lengths = output_lengths.index_select(0, restore_idx)# (batch)
        prob = F.log_softmax(logits, dim=-1)# (batch, T, num_class)
        output_dict = {}
        if txt_label is not None and txt_length is not None:
            import pdb;pdb.set_trace()
            txt_label = txt_label[txt_label.nonzero().squeeze(dim=-1)]# (sum(txt_label))
            loss = F.ctc_loss(log_probs=prob.transpose(0, 1),
                              targets=txt_label.int(),
                              input_lengths=output_lengths.int(),
                              target_lengths=txt_length.int())

            output_dict['loss'] = loss
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'acc':self.acc.get_metric(reset)}

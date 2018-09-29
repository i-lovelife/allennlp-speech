import numpy as np

from allennlp.common import Registrable

class AudioTransformer(Registrable):
    default_implementation = 'mag'
    def transform(self, fpath: str) -> np.ndarray:
        """
        Transform an audio sequence into feature.

        Returns
        -------
        features : ``np.ndarray``
        """
        raise NotImplementedError

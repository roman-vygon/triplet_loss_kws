import torch
import torch.nn.functional as F
from nemo.backends.pytorch.nm import LossNM
from nemo.core.neural_types import *
from nemo.utils.decorators import add_port_docs


class OnlineTripletLoss(LossNM):
    """
    Online Triplet loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {
            "embeds": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
            "targets": NeuralType('B', LabelsType()),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        loss:
            NeuralType(LossType)
        """
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self, margin, triplet_selector):
        super().__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def _loss(self, embeddings, target):
        embeddings = torch.flatten(embeddings, start_dim=-2)
        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean()

    def _loss_function(self, **kwargs):
        return self._loss(*(kwargs.values()))

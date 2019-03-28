import torch

from pytorch_discriminator import Classifier


c = Classifier()
c.load_state_dict(torch.load("pytorch_classifiers/0-male.pt"))

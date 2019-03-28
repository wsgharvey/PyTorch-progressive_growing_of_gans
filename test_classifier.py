import torch

from pytorch_discriminator import Classifier


c = Classifier()
c.load_state_dict("pytorch_classifiers/0-male.pt")

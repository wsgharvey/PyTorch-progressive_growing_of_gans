import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    """
    PyTorch version of discriminator from https://arxiv.org/pdf/1710.10196.pdf.
    Same as for StyleGAN discrimnator and used as classifier in StyleGAN.
    """
    def __init__(self):
        super().__init__()
        # activation func is always leaky ReLU with alpha=0.2
        self.act = lambda x: F.leaky_relu(x, negative_slope=0.2)

        # nn.ZeroPad2d(left, right, top, bottom)
        # padding calculated with script at bottom
        # apply before every convolution
        self.basic_padder = nn.ZeroPad2d((1, 1, 1, 1))
        self.strided_padder = nn.ZeroPad2d((0, 1, 0, 1))

        self.from_rgb = nn.Conv2d(3, 64, 1)                   # FromRGB_lod0
        self.conv0 = nn.Conv2d(64, 64, 3)                     # 256x256/Conv0
        self.conv0down = nn.Conv2d(64, 128, 3, stride=2)       # 256x256/Conv1_down
        self.conv1 = nn.Conv2d(128, 128, 3)                    # 128x128/Conv0
        self.conv1down = nn.Conv2d(128, 256, 3, stride=2)     # 128x128/Conv1_down
        self.conv2 = nn.Conv2d(256, 256, 3)                   # 64x64/Conv0
        self.conv2down = nn.Conv2d(256, 512, 3, stride=2)     # 64x64/Conv1_down
        self.conv3 = nn.Conv2d(512, 512, 3)                   # 32x32/Conv0
        self.conv3down = nn.Conv2d(512, 512, 3, stride=2)     # 32x32/Conv1_down
        self.conv4 = nn.Conv2d(512, 512, 3)                   # 16x16/Conv0
        self.conv4down = nn.Conv2d(512, 512, 3, stride=2)     # 16x16/Conv1_down
        self.conv5 = nn.Conv2d(512, 512, 3)                   # 8x8/Conv0
        self.conv5down = nn.Conv2d(512, 512, 3, stride=2)     # 8x8/Conv1_down
        self.conv6 = nn.Conv2d(512, 512, 3)                   # 4x4/Conv
        self.dense0 = nn.Linear(8192, 512)
        self.dense1 = nn.Linear(512, 1)

    def load_tf_weights(self, weights_dict):
        # map from names in this to names in tf model
        layer_names = {'from_rgb':  'FromRGB_lod0',
                       'conv0':     '256x256/Conv0',
                       'conv0down': '256x256/Conv1_down',
                       'conv1':     '128x128/Conv0',
                       'conv1down': '128x128/Conv1_down',
                       'conv2':     '64x64/Conv0',
                       'conv2down': '64x64/Conv1_down',
                       'conv3':     '32x32/Conv0',
                       'conv3down': '32x32/Conv1_down',
                       'conv4':     '16x16/Conv0',
                       'conv4down': '16x16/Conv1_down',
                       'conv5':     '8x8/Conv0',
                       'conv5down': '8x8/Conv1_down',
                       'conv6':     '4x4/Conv',
                       'dense0':    '4x4/Dense0',
                       'dense1':    '4x4/Dense1'}
        for own_layer_name, tf_layer_name in layer_names.items():
            own_layer = self._modules[own_layer_name]
            # load bias
            tf_bias = torch.Tensor(weights_dict[tf_layer_name+'/bias'])
            own_layer.bias = nn.Parameter(tf_bias)
            # load weights
            tf_weights = torch.Tensor(weights_dict[tf_layer_name+'/weight'])
            # permute dimensions
            if 'conv' in own_layer_name or 'rgb' in own_layer_name:
                tf_weights = tf_weights.permute(3, 2, 0, 1)
            elif 'dense' in own_layer_name:
                tf_weights = tf_weights.permute(1, 0)
            # do scaling dues to equalised learning rate stuff
            if tf_layer_name == '4x4/Dense1':
                gain = 1
            else:
                gain = torch.sqrt(torch.tensor(2.))
            tf_weights = tf_weights * gain / tf_weights[0].numel()
            # apply weight
            own_layer.weight = nn.Parameter(tf_weights)

    def forward(self, x):
        # TODO: I am ignoring minibatch norm stuff - hopefully this is right
        # TODO: need to downsample at start if using high-res images
        x = x.view(-1, 3, 256, 256)
        x = self.act(self.from_rgb(x))
        x = self.basic_padder(x)
        x = self.act(self.conv0(x))
        x = self.strided_padder(x)
        x = self.act(self.conv0down(x))
        x = self.basic_padder(x)
        x = self.act(self.conv1(x))
        x = self.strided_padder(x)
        x = self.act(self.conv1down(x))
        x = self.basic_padder(x)
        x = self.act(self.conv2(x))
        x = self.strided_padder(x)
        x = self.act(self.conv2down(x))
        x = self.basic_padder(x)
        x = self.act(self.conv3(x))
        x = self.strided_padder(x)
        x = self.act(self.conv3down(x))
        x = self.basic_padder(x)
        x = self.act(self.conv4(x))
        x = self.strided_padder(x)
        x = self.act(self.conv4down(x))
        x = self.basic_padder(x)
        x = self.act(self.conv5(x))
        x = self.strided_padder(x)
        x = self.act(self.conv5down(x))
        x = self.basic_padder(x)
        x = self.act(self.conv6(x))
        x = x.view(-1, 8192)
        x = self.act(self.dense0(x))
        x = self.act(self.dense1(x))
        return x


# # load weights from numpy files and save in PyTorch format
# import pickle
# npy_weights_dir = "/home/will/Documents/phd/research/attention/stylegan/npy-pretrained-classifiers"
# pt_weights_dir = "pytorch_classifiers"
# attributes = ['male', 'smiling', 'attractive', 'wavy-hair', 'young',
#               '5-o-clock-shadow', 'arched-eyebrows', 'bags-under-eyes', 'bald',
#               'bangs', 'big-lips', 'big-nose', 'black-hair', 'blond-hair',
#               'blurry', 'brown-hair', 'bushy-eyebrows', 'chubby',
#               'double-chin', 'eyeglasses', 'goatee', 'gray-hair',
#               'heavy-makeup', 'high-cheekbones', 'mouth-slightly-open',
#               'mustache', 'narrow-eyes', 'no-beard', 'oval-face', 'pale-skin',
#               'pointy-nose', 'receding-hairline', 'rosy-cheeks', 'sideburns',
#               'straight-hair', 'wearing-earrings', 'wearing-hat',
#               'wearing-lipstick', 'wearing-necklace', 'wearing-necktie']
# c = Classifier()
# for i, attr in enumerate(attributes):
#     weights_path = "{}/celebahq-classifier-{:0>2d}-{}.p"\
#                    .format(npy_weights_dir, i, attr)
#     save_path = "{}/{}-{}.pt".format(pt_weights_dir, i, attr)
#     weights_dict = pickle.load(open(weights_path, 'rb'))
#     c.load_tf_weights(weights_dict)
#     torch.save(c.state_dict(), save_path)
#     del weights_dict
#     print("Done {}: {}".format(i, attr))



# # padding computed from Vaibhav Dixit's answer to https://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t
# strides = (1, 1, 1)
# filter_width = filter_height = 3
# in_height = in_width = 64
# out_height = out_width = 64

# pad_along_height = max((out_height - 1) * strides[1] +
#                     filter_height - in_height, 0)
# pad_along_width = max((out_width - 1) * strides[2] +
#                    filter_width - in_width, 0)
# pad_top = pad_along_height // 2
# pad_bottom = pad_along_height - pad_top
# pad_left = pad_along_width // 2
# pad_right = pad_along_width - pad_left
# print(pad_left, pad_right, pad_top, pad_bottom)

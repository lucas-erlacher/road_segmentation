# cgan implementation (which we are currently not using at all because training is too instable)

import torch
from torch import nn, optim
from torchvision import transforms
import numpy as np
import os
import glob
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from networks.baseline_unet import UNet
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.models as models


# PARAMS
# generator
num_channels = 3
num_classes = 1  # number of "roadmaps" the unet will produce per input

# discriminator
batch_norm_momentum = 0.9

# training loop
num_training_iterats = 50
k = 4
mini_batch_size = 10
g_lr = 0.001
d_lr = 0.001


# takes in a 4D tensor (the sat_im and a roadmask stacked in the channel dimension) and returns a scalar
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # resnet
        self.layer_1 = nn.Conv2d(4, 4, 3, padding=1)
        self.layer_2 = nn.BatchNorm2d(4, momentum=batch_norm_momentum)
        self.layer_3 = nn.Conv2d(4, 3, 3)
        self.layer_4 = nn.BatchNorm2d(3, momentum=batch_norm_momentum)
        self.layer_5 = nn.MaxPool2d(2)
        self.layer_6 = nn.Conv2d(3, 3, 3, padding=1)
        self.layer_7 = nn.BatchNorm2d(3, momentum=batch_norm_momentum)
        self.layer_8 = nn.Conv2d(3, 2, 3)
        self.layer_9 = nn.BatchNorm2d(2, momentum=batch_norm_momentum)
        self.layer_10 = nn.MaxPool2d(2)
        self.layer_11 = nn.Conv2d(2, 2, 3, padding=1)
        self.layer_12 = nn.BatchNorm2d(2, momentum=batch_norm_momentum)
        self.layer_13 = nn.Conv2d(2, 1, 3)
        self.layer_14 = nn.BatchNorm2d(1, momentum=batch_norm_momentum)
        self.layer_15 = nn.MaxPool2d(2)
        # MLP
        self.layer_16 = nn.Linear(2304, 576)
        self.layer_17 = nn.Linear(576, 144)
        self.layer_18 = nn.Linear(144, 36)
        self.layer_19 = nn.Linear(36, 9)
        self.layer_20 = nn.Linear(9, 1)

    def forward(self, x):
        # resnet
        x_pre = x
        x = torch.nn.functional.relu(self.layer_1(x))
        x = self.layer_2(x)
        x = x + x_pre  # residual connection
        x = torch.nn.functional.relu(self.layer_3(x))
        x = self.layer_4(x)
        x = self.layer_5(x)
        x_pre = x
        x = torch.nn.functional.relu(self.layer_6(x))
        x = self.layer_7(x)
        x = x + x_pre  # residual connection
        x = torch.nn.functional.relu(self.layer_8(x))
        x = self.layer_9(x)
        x = self.layer_10(x)
        x_pre = x
        x = torch.nn.functional.relu(self.layer_11(x))
        x = self.layer_12(x)
        x = x + x_pre  # residual connection
        x = torch.nn.functional.relu(self.layer_13(x))
        x = self.layer_14(x)
        x = self.layer_15(x)
        # MLP
        x = torch.flatten(
            x, start_dim=1
        )  # flatten the image to prepare it for the linear layers that are coming
        x = torch.nn.functional.relu(self.layer_16(x))
        x = torch.nn.functional.relu(self.layer_17(x))
        x = torch.nn.functional.relu(self.layer_18(x))
        x = torch.nn.functional.relu(self.layer_19(x))
        x = torch.sigmoid(
            self.layer_20(x)
        )  # sigmoid in order to get a probability in the end
        return x


# custom dataloader that handles problems that could arise when requesting a new batch
# thereby allowing the client to assume that he will always receive a batch of size mini_batch_size.
class CustomLoader:
    def __init__(self, data):
        self.data = data
        self.loader = iter(data)

    # two things that could go wrong when requesting a new batch: either a StopIteration exception is thrown
    # or the returned batch is too small (bc we are less than mini_batch_size away from the end of the dataset).
    def next(self):
        try:
            x, y = next(self.loader)
        except StopIteration:
            self.restart()
            return self.restart_and_get_next()

        if len(x) < mini_batch_size:
            return self.restart_and_get_next()
        else:
            return x, y

    def restart_and_get_next(self):
        self.loader = iter(self.data)
        return self.next()


if __name__ == "__main__":
    # load data
    path = "../dataset/data/training/"
    in_prefix = "images/*"
    out_prefix = "groundtruth/*"
    in_ims = []
    out_ims = []
    convert_tensor = transforms.ToTensor()
    for file_path in glob.glob(os.path.join(path, in_prefix)):
        in_ims.append(convert_tensor(Image.open(file_path).convert("RGB")))
    for file_path in glob.glob(os.path.join(path, out_prefix)):
        out_ims.append(convert_tensor(Image.open(file_path).convert("RGB")))
    in_ims = torch.stack(in_ims)
    out_ims = torch.stack(out_ims)
    out_ims = out_ims.norm(dim=1) > 1
    out_ims = out_ims.to(torch.long)
    dataset = TensorDataset(in_ims, out_ims)
    trainloader = CustomLoader(DataLoader(dataset, batch_size=mini_batch_size))
    # training elements
    g = UNet(num_channels, num_classes)
    d = Discriminator()
    g_optimizer = optim.Adam(g.parameters(), lr=g_lr)
    d_optimizer = optim.Adam(d.parameters(), lr=d_lr)
    # training loop
    writer = SummaryWriter()
    for num_iter in range(num_training_iterats):
        print(num_iter)
        # discriminator training (k many training steps)
        for i in range(k):
            d_optimizer.zero_grad()
            mini_batch_loss = 0
            x, y = trainloader.next()
            # compute loss
            g_out = g(x)
            in_vec = torch.cat((g_out, x), 1).to(torch.float32)
            d_out_generator = d(in_vec)
            in_vec = torch.cat((y.unsqueeze(1), x), 1).to(torch.float32)
            d_out_data = d(in_vec)
            tmp_1 = torch.log(d_out_data)
            tmp_2 = torch.log(1 - d_out_generator)
            mini_batch_loss += tmp_1 + tmp_2
            # invert loss because the paper does gradient ascent (but we want to descend)
            mini_batch_loss = -mini_batch_loss
            mini_batch_loss = mini_batch_loss / mini_batch_size
            mini_batch_loss = torch.mean(mini_batch_loss, dim=0)
            d_loss = mini_batch_loss
            mini_batch_loss.backward()
            d_optimizer.step()

        # generator training (1 training step)
        g_optimizer.zero_grad()
        mini_batch_loss = 0

        x, y = trainloader.next()
        # compute loss
        g_out = g(x)
        in_vec = torch.cat((g_out, x), 1).to(torch.float32)
        d_out = d(in_vec)
        # I'm using the generator loss from the machine perception lecture (on GANs) slide 26
        mini_batch_loss += torch.log(d_out)
        mini_batch_loss = -mini_batch_loss
        mini_batch_loss = mini_batch_loss / mini_batch_size
        mini_batch_loss = torch.mean(mini_batch_loss, dim=0)
        g_loss = mini_batch_loss
        mini_batch_loss.backward()
        g_optimizer.step()

        # write to tensorboard
        writer.add_scalar("g_loss", g_loss, num_iter)
        x, y = trainloader.next()
        x = x[0]
        y = y[0]
        writer.add_image("sat_im", torchvision.utils.make_grid([x.squeeze()]), num_iter)
        writer.add_image(
            "g_out",
            torchvision.utils.make_grid([g(x.unsqueeze(0)).squeeze()]),
            num_iter,
        )
        writer.add_image("truth", torchvision.utils.make_grid([y.squeeze()]), num_iter)

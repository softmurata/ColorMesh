import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from prior_sample_dataset import TLNetSampleDataset
from prior_network import TNetwork

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_number', type=int, default=0)
parser.add_argument('--dataset_dir', type=str, default='./datasets/')
parser.add_argument('--model_dir', type=str, default='./models/')
parser.add_argument('--exp_name', type=str, default='exp1')
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--autoenc_epoch', type=int, default=100)
parser.add_argument('--enc_image_epoch', type=int, default=100)
parser.add_argument('--joint_epoch', type=int, default=100)
parser.add_argument('--voxel_size', type=int, default=20)
parser.add_argument('--v_loss_weight', type=float)

parser.add_argument('--autoenc_lr', type=float)
parser.add_argument('--encimage_lr', type=float)
parser.add_argument('--joint_lr', type=float)
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('')
args = parser.parse_args()

device = 'cuda:{}'.format(args.gpu_number) if torch.cuda.is_available() else 'cpu'

print()
print('device:', device)

# create directory which saves weight
model_dir = args.model_dir + args.exp_name + '/'
os.makedirs(model_dir, exist_ok=True)

autoenc_model_dir = model_dir + 'autoencoder/'
encimage_model_dir = model_dir + 'encimage/'
joint_model_dir = model_dir + 'joint/'

os.makedirs(autoenc_model_dir, exist_ok=True)
os.makedirs(encimage_model_dir, exist_ok=True)
os.makedirs(joint_model_dir, exist_ok=True)


dataset = TLNetSampleDataset(args)

train_dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

tnet = TNetwork(in_channels=1, hidden_channels=64, out_channels=1)

# loss function
# upper T(input voxel, output voxel) => BCELoss
# lower T(hidden_output, image_extract_output) => MSELoss

bce_loss = nn.BCELoss().to(device)
mse_loss = nn.MSELoss().to(device)

# training method
# Three stage training
# 1. autoencoder training
# 2. encoder + image extractor training
# 3. autoencoder + image extractor training

autoenc_optimizer = Adam(params=tnet.autoencoder_parameters(), lr=args.autoenc_lr, betas=(args.beta1, args.beta2))
autoenc_pretrained_weight_path = autoenc_model_dir + 'checkpoint_autoencoder.pth.tar'

print()
print('start Autoencoder training part(Step 1)')
for e in range(args.autoenc_epoch):
    losses = []

    for n, data in enumerate(train_dataloader):
        voxel, multiview_image = data
        voxel = voxel.to(device)
        multiview_image = multiview_image.to(device)

        # update autoencoder parameters
        autoenc_optimizer.zero_grad()

        _, tnet_dec_output, _ = tnet(voxel, multiview_image)

        voxel_loss = bce_loss(tnet_dec_output, voxel)

        loss = voxel_loss

        loss.backward()
        autoenc_optimizer.step()

        losses.append(loss.item())

    mean_loss = np.mean(losses)

    print('epoch: {} mean loss: {:.5f}'.format(e, mean_loss))

    # save autoencoder weight
    if e == args.autoenc_epoch - 1:

        torch.save({
            'encoder_model': tnet.encoder.state_dict(),
            'decoder_model': tnet.state_dict()
        }, autoenc_pretrained_weight_path)


print()
print('start encoder + image extractor training part(Step 2)')

encimage_optimizer = Adam(tnet.encimage_parameters(), lr=args.encimage_lr, betas=(args.beta1, args.beta2))
# load pretrained encoder model
tnet.encoder.load_state_dict(torch.load(autoenc_pretrained_weight_path)['encoder_model'])
# load pretrained image extract model
imagenet_pretrained_weight_path = ''  # ToDo: search best imagenet ML model
tnet.image_extractor.load_state_dict(torch.load(imagenet_pretrained_weight_path))
encimage_pretrained_weight_path = encimage_model_dir + 'checkpoint_encimage.pth.tar'

for e in range(args.enc_image_epoch):
    losses = []

    for n, data in enumerate(train_dataloader):
        voxel, multiview_image = data

        voxel = voxel.to(device)
        multiview_image = multiview_image.to(device)

        # update encoder + image extractor parameters
        encimage_optimizer.zero_grad()

        tnet_enc_output, _, image_extract_output = tnet(voxel, multiview_image)

        multiview_loss = mse_loss(tnet_enc_output, image_extract_output)

        loss = multiview_loss

        loss.backward()
        encimage_optimizer.step()

        losses.append(loss.item())

    mean_loss = np.mean(losses)
    print('epoch: {} mean loss: {:.5f}'.format(e, mean_loss))

    # save weight
    if e == args.enc_image_epoch - 1:
        torch.save({
            'encoder_model': tnet.encoder.state_dict(),
            'image_extractor_model': tnet.image_extractor.state_dict()
        }, encimage_pretrained_weight_path)


print()
print('start joint training part(Step 3)')

joint_optimizer = Adam(tnet.joint_parameters(), lr=args.joint_lr, betas=(args.beta1, args.beta2))
# load pretrained encoder, decoder and image extractor weight
tnet.encoder.load_state_dict(torch.load(encimage_pretrained_weight_path['encoder_model']))
tnet.decoder.load_state_dict(torch.load(autoenc_pretrained_weight_path['decoder_model']))
tnet.image_extractor.load_state_dict(torch.load(encimage_pretrained_weight_path['image_extractor_model']))



for e in range(args.epoch):
    losses = []

    for n, data in enumerate(train_dataloader):
        voxel, multiview_image = data

        voxel = voxel.to(device)
        multiview_image = multiview_image.to(device)

        joint_optimizer.zero_grad()

        tnet_enc_output, tnet_dec_output, image_extract_output = tnet(voxel, multiview_image)
        # encoder output => (batch_size, hidden_channels)
        # decoder output => (batch_size, voxel_size * voxel_size * voxel_size) or (batch_size, voxel_size, voxel_size, voxel_size)
        # image extract output => (batch_size, hidden_channels)

        # create loss function
        voxel_loss = bce_loss(tnet_dec_output, voxel)
        multiview_loss = mse_loss(tnet_enc_output, image_extract_output)

        loss = voxel_loss + multiview_loss

        loss.backward()
        joint_optimizer.step()

        losses.append(loss.item())



    if e % args.save_freq == 0:
        model_weight_path = joint_model_dir + 'checkpoint_prior%05d.pth.tar' % e

        torch.save({
            'encoder_model': tnet.encoder.state_dict(),
            'decoder_model': tnet.decoder.state_dict(),
            'image_extract_model': tnet.image_extractor.state_dict()
        }, model_weight_path)





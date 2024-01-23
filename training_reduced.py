import torch
import torch.nn as nn
import argparse
from pathlib import Path
from torchvision.transforms import v2
import os

from util.train import train
from util.test import test
from reduced_models.resnet import resnet_reduced, resnet_nostem, resnet_small
from reduced_models.resnext import ResNext, ResNext_nostem
from reduced_models.convnext import ConvNeXt, ConvNeXt_kernel7, ConvNeXt_nostem, ConvNeXtkernel7_nostem, ConvNeXtkernel7_increasedim, ConvNeXtkernel7_increasedim_nostem
from util.cifar100_data_loader import load_data
from util.miniImagenet_data_loader import load_miniImagenet


def get_args_parser():
    parser = argparse.ArgumentParser('ConvNeXt training and evaluation script for image classification', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--update_freq', default=1, type=int,
                        help='gradient accumulation steps')

    # Model parameters
    parser.add_argument('--model', default=None,
                        choices=['resnet_reduced', 'resnet_nostem', 'resnet_small', 'ResNext', 'ResNext_nostem',
                                 'ConvNeXt', 'ConvNext_kernel7', 'ConvNeXt_nostem', 'ConvNeXtkernel7_nostem',
                                 'ConvNeXtkernel7_increasedim', 'convnext_microdesign'],
                        type=str, help='ImageNet dataset path, choices=[resnet_reduced, resnet_nostem, resnet50_small, '
                                       'ResNext, ResNextCifar, ConvNeXt, ConvNext_kernel7, ConvNeXt_nostem, '
                                       'ConvNeXtkernel7_nostem, ConvNeXtkernel7_increasedim, convnext_microdesign]')
    parser.add_argument('--input_size', default=84, type=int,
                        help='image input size')

    # Optimization parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0, metavar='M',
                        help='SGD momentum (default: 0)')
    parser.add_argument('--weight_decay', type=float, default=1e-8,
                        help='weight decay (default: 1e-8)')

    parser.add_argument('--lr', type=float, default=1e-7, metavar='LR',
                        help='learning rate (default: 1e-7)')

    # Dataset

    parser.add_argument('--data_set', default='Mini-Imagenet', choices=['CIFAR', 'Mini-Imagenet'],
                        type=str, help='ImageNet dataset path, choices=[CIFAR, Mini-Imagenet]')
    parser.add_argument('--output_dir', default='',
                        help='path where to save model results, empty for no saving')

    return parser


def main(args):

    torch.manual_seed(123)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    if args.data_set == 'CIFAR':
        trainloader, valloader, testloader = load_data(args.batch_size)
    else:
        trainloader, valloader, testloader = load_miniImagenet(args.batch_size)

    cutmix = v2.CutMix(num_classes=100)
    mixup = v2.MixUp(num_classes=100)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

    for images, labels in trainloader:
        images, labels = cutmix_or_mixup(images, labels)

    net = None
    if args.model == 'resnet_reduced':
        net = resnet_reduced(100)
    elif args.model == 'resnet_nostem':
        net = resnet_nostem(100)
    elif args.model == 'resnet_small':
        net = resnet_small(100)
    elif args.model == 'ResNext':
        net = ResNext(100)
    elif args.model == 'ResNext_nostem':
        net = ResNext_nostem(100)
    elif args.model == 'ConvNeXt':
        net = ConvNeXt(100)
    elif args.model == 'ConvNeXt_kernel7':
        net = ConvNeXt_kernel7(100)
    elif args.model == 'ConvNeXt_nostem':
        net = ConvNeXt_nostem(100)
    elif args.model == 'ConvNeXtkernel7_nostem':
        net = ConvNeXtkernel7_nostem(100)
    elif args.model == 'ConvNeXtkernel7_increasedim':
        net = ConvNeXtkernel7_increasedim(100)
    elif args.model == 'ConvNeXtkernel7_increasedim_nostem':
        net = ConvNeXtkernel7_increasedim_nostem(100)

    print(net)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(trainloader) * args.epochs)

    train_losses = []
    val_losses = []
    val_accuracies = []

    test(net, device, valloader, criterion, True)
    for epoch in range(1, args.epochs + 1):
        train_loss = train(net, device, trainloader, criterion, optimizer, epoch)
        train_losses.append(train_loss)
        val_loss, val_acc = test(net, device, valloader, criterion, True)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        scheduler.step()
    if args.output_dir != '':
        path = './' + args.output_dir + '/' + args.model + '.h5'
        torch.save(net.state_dict(), path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ConvNeXt reduced models training script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

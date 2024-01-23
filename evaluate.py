import torch
from ptflops import get_model_complexity_info
import argparse

from reduced_models.resnet import resnet_reduced, resnet_nostem, resnet_small
from reduced_models.resnext import ResNext, ResNext_nostem
from reduced_models.convnext import ConvNeXt, ConvNeXt_kernel7, ConvNeXt_nostem, ConvNeXtkernel7_nostem, \
    ConvNeXtkernel7_increasedim, ConvNeXtkernel7_increasedim_nostem
from util.cifar100_data_loader import load_data
from util.miniImagenet_data_loader import load_miniImagenet
from util.test import evaluate


def get_args_parser():
    parser = argparse.ArgumentParser('ConvNeXt training and evaluation script for image classification', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Per GPU batch size')

    # Model parameters
    parser.add_argument('--model', default=None,
                        choices=['resnet_reduced', 'resnet_nostem', 'resnet_small', 'ResNext', 'ResNext_nostem',
                                 'ConvNeXt', 'ConvNext_kernel7', 'ConvNeXt_nostem', 'ConvNeXtkernel7_nostem',
                                 'ConvNeXtkernel7_increasedim', 'convnext_microdesign'],
                        type=str, help='ImageNet dataset path, choices=[resnet_reduced, resnet_nostem, resnet50_small, '
                                       'ResNext, ResNextCifar, ConvNeXt, ConvNext_kernel7, ConvNeXt_nostem, '
                                       'ConvNeXtkernel7_nostem, ConvNeXtkernel7_increasedim, convnext_microdesign]')

    parser.add_argument('--data_set', default='Mini-Imagenet', choices=['CIFAR', 'Mini-Imagenet'],
                        type=str, help='ImageNet dataset path, choices=[CIFAR, Mini-Imagenet]')
    parser.add_argument('--weights_dir', default='',
                        help='path of the trained weights', required=True)

    return parser


def main(args):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    if args.data_set == 'CIFAR':
        trainloader, valloader, testloader = load_data(args.batch_size)
        input_size = 32
    else:
        trainloader, valloader, testloader = load_miniImagenet(args.batch_size)
        input_size = 84



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

    print('\n' + args.model)
    macs, params = get_model_complexity_info(net, (3, input_size, input_size), as_strings=True,
                                             print_per_layer_stat=False, verbose=False)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    weights = torch.load(args.weights_dir)
    net.load_state_dict(weights)
    net.to(device)

    evaluate(net, device, testloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ConvNeXt evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

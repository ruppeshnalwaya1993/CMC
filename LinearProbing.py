from __future__ import print_function

import os, copy
import sys
import time
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.distributed as dist
import argparse
import socket
from torch.utils.data import distributed
import tensorboard_logger as tb_logger

from torchvision import transforms, datasets
import torchvision.models as models
from dataset import RGB2Lab
from util import adjust_learning_rate, AverageMeter, accuracy

from models.alexnet import alexnet
from models.resnet import ResNetV2
from models.LinearModel import LinearClassifierAlexNet, LinearClassifierResNetV2
from caffenet import CaffeNet_BN

from spawn import spawn

from VideoLoader import VideoLoader, MyVideoFolder
import spatial_transforms as VT
from mean import get_mean, get_std

def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=5, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=32, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,200', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')

    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    # model definition
    parser.add_argument('--model', type=str, default='alexnet', choices=['alexnet', 'resnet50', 'resnet101', 'caffenet'])
    parser.add_argument('--model_path', type=str, default=None, help='the model to test')
    parser.add_argument('--layer', type=int, default=5, help='which layer to evaluate')

    # path definition
    parser.add_argument('--data_folder', type=str, default=None, help='path to data')
    parser.add_argument('--save_path', type=str, default=None, help='path to save linear classifier')
    parser.add_argument('--tb_path', type=str, default=None, help='path to tensorboard')

    # data crop threshold
    parser.add_argument('--crop_low', type=float, default=0.08, help='low area in crop')

    # log file
    parser.add_argument('--log', type=str, default='time_linear.txt', help='log file')

    # parallel setting
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    parser.add_argument('--scratch', action='store_true', help='training from random weights')
    parser.add_argument('--ten_crop', action='store_true', help='use ten crops for validation')
    parser.add_argument('--evaluate', action='store_true', help='vaidation only once')

    opt = parser.parse_args()

    if (opt.data_folder is None) or (opt.save_path is None) or (opt.tb_path is None):
        raise ValueError('one or more of the folders is None: data_folder | save_path | tb_path')

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = opt.model_path.split('/')[-2]
    opt.model_name = 'calibrated_{}_bsz_{}_lr_{}_decay_{}'.format(opt.model_name, opt.batch_size, opt.learning_rate,
                                                                  opt.weight_decay)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name + '_layer{}'.format(opt.layer))
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.save_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def get_new_features(model, conv, arch):
    print(arch)
    children = list(model.features.children())
    count = 1
    for i,m in enumerate(children):
        if (arch == 'vgg16' and isinstance(m, nn.MaxPool2d)) or (arch == 'alexnet' and isinstance(m, nn.ReLU)):
            if count == conv:
                break
            count = count + 1
    if arch == 'alexnet':
        i += 2
    return nn.Sequential(*list(model.features.children())[:i])


def input_transformation_no_diff(x):
    x = torch.transpose(x, 0, 1)
    return x


def get_train_val_loader(args):
    train_folder = os.path.join(args.data_folder, 'train')
    val_folder = os.path.join(args.data_folder, 'val')

    normalize = VT.GroupToNormalizedTensor(mean=get_mean(), std=get_std())
    transform = [
            VT.GroupResize(256),
            VT.GroupRandomCrop(224),
            VT.GroupRandomHorizontalFlip(),
            VT.GroupToBGR2RGB(),
            VT.GroupColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            normalize,
            VT.GroupFormat()]
    transform += [transforms.Lambda(lambda x: input_transformation_no_diff(x))]
    transform = VT.Compose(transform)

    video_loader = VideoLoader(
            num_frames=1,
            step_size=1,
            diff_stride=0,
            samples_per_video=1,
            random_offset=True)
    train_dataset = MyVideoFolder(
                train_folder,
                transform=transform,
                loader=video_loader,
                error_paths=[])


    transform = [
            VT.GroupResize(256),
            VT.GroupCenterCrop(224),
            VT.GroupToBGR2RGB(),
            normalize,
            VT.GroupFormat(),
            transforms.Lambda(lambda x: torch.transpose(x, 0, 1))]
    if args.ten_crop:
        grp_rgb2bgr = VT.GroupToBGR2RGB()
        grp_resize = VT.GroupResize(256)
        grp_tencrop = VT.GroupTenCrop(224)
        transform = [
            transforms.Lambda(lambda clips_list: [ grp_resize(clip) for clip in clips_list ]),
            transforms.Lambda(lambda clips_list: [ grp_tencrop(clip) for clip in clips_list ]),
            transforms.Lambda(lambda clipsclips_list: [clip for clips_list in clipsclips_list for clip in clips_list]),
            transforms.Lambda(lambda crops: [grp_rgb2bgr(crop) for crop in crops]),
            transforms.Lambda(lambda crops: [normalize(crop) for crop in crops]),
            VT.GroupFormat(),
            transforms.Lambda(lambda x: torch.transpose(x, 1, 2))]
    transform = VT.Compose(transform)

    if args.ten_crop:
        video_loader = VideoLoader(
            num_frames=1,
            step_size=1,
            diff_stride=0,
            samples_per_video=25,
            random_offset=False)
    else:
        video_loader = VideoLoader(
            num_frames=1,
            step_size=1,
            diff_stride=0,
            samples_per_video=1,
            random_offset=False)
    val_dataset = MyVideoFolder(
                val_folder,
                transform=transform,
                loader=video_loader,
                error_paths=[])


    print('number of train: {}'.format(len(train_dataset)))
    print('number of val: {}'.format(len(val_dataset)))

    if args.distributed:
        # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_sampler = distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)

    bsz = args.batch_size
    if args.ten_crop:
        bsz = 4
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=bsz, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    return train_loader, val_loader, train_sampler


def set_model(args, ngpus_per_node):
    if args.model == 'alexnet':
        model = models.__dict__['alexnet'](num_classes=128)
        model2 = models.__dict__['alexnet'](num_classes=101)
        classifier = copy.deepcopy(model2.classifier) #LinearClassifierAlexNet(layer=args.layer, n_label=101, pool_type='max')
    elif args.model.startswith('resnet'):
        model = ResNetV2(args.model)
        classifier = LinearClassifierResNetV2(layer=args.layer, n_label=1000, pool_type='avg')
    elif args.model == 'caffenet':
        model = CaffeNet_BN(output_layer='fc8', num_classes=101)
    else:
        raise NotImplementedError(args.model)

    # load pre-trained model
    if not args.scratch:
        print('==> loading pre-trained model')
        ckpt = torch.load(args.model_path)
        state_dict = ckpt['model']

        has_module = False
        for k, v in state_dict.items():
            if k.startswith('module'):
                has_module = True

        if has_module:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)

    if args.model == 'alexnet':
        #model.features = get_new_features(model, args.layer, 'alexnet')
        model.classifier = classifier
    if args.model == 'caffenet' and not args.scratch and args.resume == '':
        model.reinit_fc()

    print('==> done')
    model.eval()
    print(model)

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.num_workers = int(args.num_workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    return model, criterion


def set_optimizer(args, classifier):
    optimizer = optim.SGD(classifier.parameters(),
                          lr=args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    return optimizer


def train(epoch, train_loader, model, criterion, optimizer, opt):
    """
    one epoch training
    """
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, ele in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        ((input, _), target), _ = ele
        input = input.float()
        if opt.gpu is not None:
            input = input.cuda(opt.gpu, non_blocking=True)
        target = target.cuda(opt.gpu, non_blocking=True)

        input = torch.reshape(input, (input.shape[0] * input.shape[1], input.shape[2], input.shape[3], input.shape[4]))

        # ===================forward=====================
        #with torch.no_grad():
        #    output = model(input)
        #    feat = feat.contiguous().detach()

        output = model(input)#classifier(feat)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # tensorboard logger
        pass

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    return top1.avg, losses.avg


def validate(val_loader, model, criterion, opt):
    """
    evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, ele in enumerate(val_loader):
            ((input, _), target), _ = ele
            input = input.float()
            if opt.gpu is not None:
                input = input.cuda(opt.gpu, non_blocking=True)
            target = target.cuda(opt.gpu, non_blocking=True)
            bsz = input.shape[0]
            if opt.ten_crop:
                num_frames = input.shape[2]
                num_crops = input.shape[1]
                input = torch.reshape(input, (bsz * num_crops * num_frames, input.shape[3], input.shape[4], input.shape[5]))
            else:
                num_frames = input.shape[1]
                input = torch.reshape(input, (bsz * num_frames, input.shape[2], input.shape[3], input.shape[4]))

            # compute output
            #feat = model(input)
            #feat = feat.contiguous().detach()
            output = model(input)#classifier(feat)
            if opt.ten_crop:
                output = torch.mean(torch.reshape(output, (bsz, num_crops*num_frames, output.shape[-1])), dim=1)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, losses.avg


def main():
    args = parse_option()
    print(args)

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()

    with open(args.log, 'w') as f:
        pass

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    best_acc1 = 0

    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # set the model
    model, criterion = set_model(args, ngpus_per_node)

    # set optimizer
    optimizer = set_optimizer(args, model)

    cudnn.benchmark = True

    # optionally resume model
    args.start_epoch = 1
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] + 1
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            sys.exit(0)

    # set the data loader
    train_loader, val_loader, train_sampler = get_train_val_loader(args)

    # tensorboard
    logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)

    if args.evaluate:
        print("==> testing...")
        test_acc, test_loss = validate(val_loader, model, criterion, args)
        return

    # routine
    for epoch in range(args.start_epoch, args.epochs + 1):

        if args.distributed:
            train_sampler.set_epoch(epoch)

        adjust_learning_rate(epoch, args, optimizer)
        print("==> training...")

        time1 = time.time()
        train_acc, train_loss = train(epoch, train_loader, model, criterion, optimizer, args)
        time2 = time.time()
        print('train epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)

        print("==> testing...")
        test_acc, test_loss = validate(val_loader, model, criterion, args)

        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_loss', test_loss, epoch)

        # save the best model
        if test_acc > best_acc1:
            best_acc1 = test_acc
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                        and args.rank % ngpus_per_node == 0):
                state = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                }
                save_name = '{}_layer{}.pth'.format(args.model, args.layer)
                save_name = os.path.join(args.save_folder, save_name)
                print('saving model!')
                torch.save(state, save_name)

        # regular save
        if not args.multiprocessing_distributed or \
                (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            if epoch % args.save_freq == 0:
                print('==> Saving...')
                state = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                }
                save_file = os.path.join(args.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
                torch.save(state, save_file)

        # tensorboard logger
        pass


if __name__ == '__main__':
    best_acc1 = 0
    main()

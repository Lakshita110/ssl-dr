import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import models
import random
from lib.LinearAverage import LinearAverage
from lib.BatchAverage import BatchCriterion
from lib.BatchAverageRot import BatchCriterionRot
from lib.utils import AverageMeter
from test import kNN
import numpy as np

from lib.utils import save_checkpoint, adjust_learning_rate, accuracy
from lib.utils import get_color_distortion, gaussian_blur
from tensorboardX import SummaryWriter

model_names = sorted(name for name in models.__dict__ if name.islower()
                     and not name.startswith("__") 
                     and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--low-dim', default=128, type=int,
                    metavar='D', help='feature dimension')
parser.add_argument('--nce-k', default=4096, type=int,
                    metavar='K', help='number of negative samples for NCE')
parser.add_argument('--nce-t', default=0.07, type=float,
                    metavar='T', help='temperature parameter for softmax')
parser.add_argument('--nce-m', default=0.5, type=float,
                    help='momentum for non-parametric updates')
parser.add_argument('--iter_size', default=1, type=int,
                    help='caffe style iter size')

parser.add_argument('--result', default="", type=str)
parser.add_argument('--seedstart', default=0, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--seedend', default=5, type=int)


parser.add_argument("--synthesis", action="store_true")
parser.add_argument('--showfeature', action="store_true")
parser.add_argument('--multiaug', action="store_true")
parser.add_argument('--multitask', action="store_true")
parser.add_argument('--domain', action="store_true")
parser.add_argument("--saveembed", type=str, default="")

best_prec1 = 0

def main():

    global args, best_prec1
    args = parser.parse_args()
    
    print("args.multitask", args.multitask)

    my_whole_seed = 111
    random.seed(my_whole_seed)
    np.random.seed(my_whole_seed)
    torch.manual_seed(my_whole_seed)
    torch.cuda.manual_seed_all(my_whole_seed)
    torch.cuda.manual_seed(my_whole_seed)
    np.random.seed(my_whole_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(my_whole_seed)

    for kk_time in range(args.seedstart, args.seedend):
        args.seed = kk_time
        args.result = args.result + str(args.seed)

        # create model
        model = models.__dict__[args.arch](low_dim=args.low_dim, multitask=args.multitask , showfeature=args.showfeature, args = args)
        #
        # from models.Gresnet import ResNet18
        # model = ResNet18(low_dim=args.low_dim, multitask=args.multitask)
        model = torch.nn.DataParallel(model).cuda()

        # Data loading code
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        aug = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                                  transforms.RandomGrayscale(p=0.2),
                                  transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  normalize])
        # aug = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.08, 1.), ratio=(3 / 4, 4 / 3)),
        #                           transforms.RandomHorizontalFlip(p=0.5),
        #                           get_color_distortion(s=1),
        #                           transforms.Lambda(lambda x: gaussian_blur(x)),
        #                           transforms.ToTensor(),
        #                           normalize])
        # aug = transforms.Compose([transforms.RandomRotation(60),
        #                           transforms.RandomResizedCrop(224, scale=(0.6, 1.)),
        #                           transforms.RandomGrayscale(p=0.2),
        #                           transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        #                           transforms.RandomHorizontalFlip(),
        #                           transforms.ToTensor(),
        #                             normalize])
        aug_test = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                normalize])

        # dataset
        import datasets.fundus_kaggle_dr as medicaldata
        train_dataset = medicaldata.traindataset(root=args.data, transform=aug, train=True, args=args)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2, drop_last=True if args.multiaug else False,  worker_init_fn=random.seed(my_whole_seed), persistent_workers=True)

        valid_dataset = medicaldata.traindataset(root="/cluster/tufts/cs152l3dclass/areddy05/IDRID/Images", transform=aug_test, train=False, args=args)
        val_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2, worker_init_fn=random.seed(my_whole_seed))

        # define lemniscate and loss function (criterion)
        ndata = train_dataset.__len__()

        lemniscate = LinearAverage(args.low_dim, ndata, args.nce_t, args.nce_m).cuda()
        local_lemniscate = None

        if args.multitask:
            print ("running multi task with positive")
            criterion = BatchCriterionRot(1, 0.1, args.batch_size, args).cuda()
        elif args.domain:
            print ("running domain with four types--unify ")
            from lib.BatchAverageFour import BatchCriterionFour
            # criterion = BatchCriterionTriple(1, 0.1, args.batch_size, args).cuda()
            criterion = BatchCriterionFour(1, 0.1, args.batch_size, args).cuda()
        elif args.multiaug:
            print ("running multi task")
            criterion = BatchCriterion(1, 0.1, args.batch_size, args).cuda()
        else:
            criterion = nn.CrossEntropyLoss().cuda()

        if args.multitask:
            cls_criterion = nn.CrossEntropyLoss().cuda()
        else:
            cls_criterion = None

        optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                     weight_decay=args.weight_decay)

        if args.evaluate:
            knn_num = 100
            auc, acc, precision, recall, f1score = kNN(args, model, lemniscate, train_loader, val_loader, knn_num, args.nce_t, 2)
            f = open("savemodels/result.txt", "a+")
            f.write("auc: %.4f\n" % (auc))
            f.write("acc: %.4f\n" % (acc))
            f.write("pre: %.4f\n" % (precision))
            f.write("recall: %.4f\n" % (recall))
            f.write("f1score: %.4f\n" % (f1score))
            f.close()
            return

        # mkdir result folder and tensorboard
        os.makedirs(args.result, exist_ok=True)
        writer = SummaryWriter("runs/" + str(args.result.split("/")[-1]))
        writer.add_text('Text', str(args))

        for epoch in range(args.start_epoch, args.epochs):
            lr = adjust_learning_rate(optimizer, epoch, args, [100, 200])
            writer.add_scalar("lr", lr, epoch)

            # train for one epoch
            print(args.multitask)
            loss = train(train_loader, model, lemniscate, local_lemniscate, criterion, cls_criterion, optimizer, epoch, writer)
            writer.add_scalar("train_loss", loss, epoch)

            # gap_int = 10
            # if (epoch) % gap_int == 0:
            #     knn_num = 100
            #     auc, acc, precision, recall, f1score = kNN(args, model, lemniscate, train_loader, val_loader, knn_num, args.nce_t, 2)
            #     writer.add_scalar("test_auc", auc, epoch)
            #     writer.add_scalar("test_acc", acc, epoch)
            #     writer.add_scalar("test_precision", precision, epoch)
            #     writer.add_scalar("test_recall", recall, epoch)
            #     writer.add_scalar("test_f1score", f1score, epoch)
            #
            #     auc, acc, precision, recall, f1score = kNN(args, model, lemniscate, train_loader, val_loader_gon,
            #                                                knn_num, args.nce_t, 2)
            #     writer.add_scalar("gon/test_auc", auc, epoch)
            #     writer.add_scalar("gon/test_acc", acc, epoch)
            #     writer.add_scalar("gon/test_precision", precision, epoch)
            #     writer.add_scalar("gon/test_recall", recall, epoch)
            #     writer.add_scalar("gon/test_f1score", f1score, epoch)
            #     auc, acc, precision, recall, f1score = kNN(args, model, lemniscate, train_loader, val_loader_pm,
            #                                                knn_num, args.nce_t, 2)
            #     writer.add_scalar("pm/test_auc", auc, epoch)
            #     writer.add_scalar("pm/test_acc", acc, epoch)
            #     writer.add_scalar("pm/test_precision", precision, epoch)
            #     writer.add_scalar("pm/test_recall", recall, epoch)
            #     writer.add_scalar("pm/test_f1score", f1score, epoch)

            # save checkpoint
            save_checkpoint({
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'lemniscate': lemniscate,
                'optimizer': optimizer.state_dict(),
            }, filename=args.result + "/fold" + str(args.seedstart) + "-epoch-" + str(epoch) + ".pth.tar")


def train(train_loader, model, lemniscate, local_lemniscate, criterion, cls_criterion, optimizer, epoch, writer):
    print("train")
    print("args.multitask", args.multitask)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_ins = AverageMeter()
    losses_rot = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    optimizer.zero_grad()

    for i, (input, target, index, name) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        if args.multitask:
            # Check if input is a list (multiple augmentations) or single tensor
            if isinstance(input, list):
                # If input is a list of tensors, concatenate them
                input = torch.cat([aug.cuda() for aug in input], 0)
                index = torch.cat([index, index], 0).cuda()
                rotation_label = torch.cat([target[1], target[1]], 0).cuda()
            else:
                # If input is a single tensor, just move it to cuda
                input = input.cuda()
                index = index.cuda()
                rotation_label = target[1].cuda() if isinstance(target, list) else target.cuda()

            # initialize tensors for rotations
            current_batch_size = input.size(0)
            
            # construct rotated versions
            dataX_90 = torch.flip(torch.transpose(input, 2, 3), [2])
            dataX_180 = torch.flip(torch.flip(input, [2]), [3])
            dataX_270 = torch.transpose(torch.flip(input, [2]), 2, 3)
            
            # Stack all rotated versions
            dataX = torch.cat([input, dataX_90, dataX_180, dataX_270], dim=0)
            
            # Create rotation labels (0,1,2,3 for 0,90,180,270 degrees)
            rotation_labels = torch.cat([
                torch.zeros(current_batch_size),
                torch.ones(current_batch_size),
                2 * torch.ones(current_batch_size),
                3 * torch.ones(current_batch_size)
            ]).long().cuda()
            
            # Repeat indices for each rotation
            indices = torch.cat([index] * 4)

            # Forward pass
            feature, pred_rot, feature_whole = model(dataX)

            # Compute losses
            loss_instance = criterion(feature, indices) / args.iter_size
            loss_cls = cls_criterion(pred_rot, rotation_labels)
            loss = loss_instance + 1.0 * loss_cls  # 1.0 is the weight for rotation loss

            losses_ins.update(loss_instance.item() * args.iter_size, input.size(0))
            losses_rot.update(loss_cls.item() * args.iter_size, input.size(0))

        else:
            print("running single task")
            input = input.cuda()
            index = index.cuda()

            feature = model(input)
            output = lemniscate(feature, index)
            loss = criterion(output, index) / args.iter_size

        # backward pass
        loss.backward()

        # measure accuracy and record loss
        losses.update(loss.item() * args.iter_size, input.size(0))

        if (i+1) % args.iter_size == 0:
            # compute gradient and do optimization step
            optimizer.step()
            optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))

    writer.add_scalar("losses_ins", losses_ins.avg, epoch)
    writer.add_scalar("losses_rot", losses_rot.avg, epoch)

    return losses.avg

if __name__ == '__main__':
    main()
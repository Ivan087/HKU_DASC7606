import argparse
import collections
import os
import numpy as np
import random
from tqdm import tqdm
import logging
from torch.utils.tensorboard import SummaryWriter


import torch
import torch.optim as optim
from torchvision import transforms
from torchvision.transforms import RandomGrayscale

from retinanet import model
from retinanet.dataloader import CocoDataset, collater, Resizer, \
    AspectRatioBasedSampler, Augmenter, Normalizer
from retinanet.eval import Evaluation
    
from torch.utils.data import DataLoader, Subset

def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--coco_path', help='Path to COCO directory', default='./data')
    parser.add_argument('--output_path', help='Path to output directory to save checkpoints', default='./output')
    parser.add_argument('--depth', help='ResNet depth, must be one of 18, 34, 50, 101, 152', type=int, default=101)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=72)
    parser.add_argument('--logs', help='Path of log output',type=str, default='./output/log_file.txt')
    parser.add_argument('--batch_size', help='Batch size of detection samples', type=int, default=2)
    parser.add_argument('--num_workers', help='Number of cpu workers to work parallel', type=int, default=3)
    parser = parser.parse_args(args)

    # Tensorboard
    writer = SummaryWriter(comment ='depth_{}_epoch_{}'.format(parser.depth,parser.epochs))

    if not os.path.exists(parser.output_path):
        os.mkdir(parser.output_path)

    # log output
    logger = logging.getLogger()         
    logger.setLevel(logging.INFO)        

    file_handler = logging.FileHandler(parser.logs)   
    console_handler = logging.StreamHandler()       

    # 
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s "
    )

    file_handler.setFormatter(formatter)       
    console_handler.setFormatter(formatter)    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    if parser.coco_path is None:
        raise ValueError('Must provide --coco_path when training on COCO.')

    logger.info("{}Loading dataset ....{}".format('#'*10,'#'*10))
    dataset_train = CocoDataset(parser.coco_path, set_name='train',
                                transform=transforms.Compose([ Augmenter(),
                                                                # RandomGrayscale(),
                                                               Normalizer(),
                                                                Resizer()]))
    dataset_val = CocoDataset(parser.coco_path, set_name='val',
                                transform=transforms.Compose([Normalizer(), Resizer()]))

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=parser.batch_size, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=parser.num_workers, collate_fn=collater, batch_sampler=sampler)

    logger.info("{}Loading model ....{}".format('#'*10,'#'*10))
    # Create the model
    if parser.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 18:
        retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
    else:
        raise ValueError('Unsupported model depth')

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    retinanet.training = True
    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[48, 64])
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5, last_epoch=-1)
    loss_hist = collections.deque(maxlen=500)
    epoch_loss_list = []


    logger.info('Num training images: {}'.format(len(dataset_train)))

    for epoch_num in range(parser.epochs):
        
        retinanet.training = True
        retinanet.train()
        retinanet.freeze_bn()

        epoch_loss = []

        for iter_num, data in tqdm(enumerate(dataloader_train)):
            
            ###################################################################
            # TODO: Please fill the codes here to zero optimizer gradients
            ##################################################################
            optimizer.zero_grad()

            ##################################################################

            if torch.cuda.is_available():
                classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot'].cuda()])
            else:
                classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])
                
            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()
            
            loss = classification_loss + regression_loss


            if bool(loss == 0):
                continue

            ###################################################################
            # TODO: Please fill the codes here to complete the gradient backward
            ##################################################################
            loss.backward()

            ##################################################################

            torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

            ###################################################################
            # TODO: Please fill the codes here to optimize parameters
            ##################################################################
            optimizer.step()

            ##################################################################

            loss_hist.append(float(loss))

            epoch_loss.append(float(loss))

            if iter_num % 100 == 0:
                # print(
                #     'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                #         epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

                logger.info(
                    '\nEpoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                        epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))
                

            # Tensorboard record
            if iter_num % 500 == 0:
                writer.add_scalar("Classification_Loss/train", classification_loss, epoch_num*1000 +iter_num)
                writer.add_scalar("Regression_Loss/train", regression_loss, epoch_num*1000 +iter_num)
                writer.add_scalar("Total_Loss/train", np.mean(loss_hist), epoch_num*1000 +iter_num)
                writer.add_scalar("Iteration_Loss/train", loss_hist[-1], epoch_num*1000 +iter_num)

                # validation sample size
                # subset_size = 200

                # index for validation set
                # indices = torch.randperm(len(dataset_val))[:subset_size].tolist()
                # start_index = min(indices)
                # end_index = max(indices) + 1
                # slice_index = slice(start_index, end_index) 

                # dataset_size = len(dataset_val)

                # sample set of validation set
                # subset = Subset(dataset_val, indices)
                # print('type of val{}'.format(type(dataset_val)))
                # print('type of sample{}'.format(type(subset)))
                # retinanet.eval()
                # retinanet.training = False
                # eval = Evaluation()
                # # subset = dataset_val[slice_index]
                # # print('type of sample{}'.format(type(subset)))
                # # print('type of sample{}'.format(type(subset[0])))
                # mAP_result = eval.evaluate(dataset_val, retinanet)


                # writer.add_scalars('Fitting process', {'Validation':mAP_result[0],'Train':loss}, iter_num)

            del classification_loss
            del regression_loss

        scheduler.step()

        epoch_loss_list.append(np.mean(epoch_loss))

        if (epoch_num + 1) % 3 == 0 or epoch_num + 1 == parser.epochs:
            logger.info('Evaluating dataset')
            retinanet.eval()
            retinanet.training = False
            eval = Evaluation()
            mAP_result = eval.evaluate(dataset_val, retinanet)
            logger.info('mAP result: {}'.format(mAP_result))
            writer.add_scalars('Fitting process', {'Validation':mAP_result[0],'Train':epoch_loss_list[-1]}, epoch_num)
            if (epoch_num + 1) % 6 == 0 or epoch_num + 1 == parser.epochs:
                torch.save(retinanet, os.path.join(parser.output_path, 'retinanet_epoch{}.pt'.format(epoch_num + 1)))
                logger.info('Model saved with epoch {}'.format(epoch_num))
    writer.flush()
    writer.close()

    torch.save(retinanet, os.path.join(parser.output_path, 'model_final.pt'))


if __name__ == '__main__':
    main()

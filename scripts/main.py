# Import packages
import warnings
warnings.filterwarnings("ignore")
# Pytorch Deep Learning packages
import torch
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import tensorboardX
# Import system packages
import os
import argparse
import parser
# Import function from user defined file
from utils.function import time2str, build_dirs, set_parameter_requires_grad
import scripts.configuration as conf
from os.path import join
from dataset.dataset import Dataset
from models.WSCNet import ClassWisePool, ResNetWSL

# Parameter setting
parser = argparse.ArgumentParser(description='Video Sentiment')
parser.add_argument('--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=conf.NUM_EPOCHS, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=conf.BATCH_SIZE,
                    type=int, metavar='N',
                    help='mini-batch size (default: {})'.format(conf.BATCH_SIZE), dest='batch_size')
parser.add_argument('--lr', '--learning-rate', default=conf.LEARNING_RATE,
                    type=float, metavar='LR', help='initial learning rate',
                    dest='learning_rate')
parser.add_argument('--num_feature', default=conf.NUM_OF_FEAT,
                    type=int, metavar='N', help='initial number of feature',
                    dest='num_feature')
parser.add_argument('--num_class', default=conf.NUM_OF_CLASS,
                    type=int, metavar='N', help='initial number of class',
                    dest='num_class')
parser.add_argument('--input_size', default=conf.IMAGE_SIZE,
                    type=int, metavar='N', help='initial input size',
                    dest='input_size')
parser.add_argument('--feature_extract', default=conf.FINTUNE_FC,
                    type=bool, metavar='N', help='fintune all or not',
                    dest='feature_extract')


def main():
    args = parser.parse_args()

    # initializ gloabl path
    global_path = os.path.dirname(os.path.realpath(__file__))
    conf.global_path = global_path
    print('the global path: '.format(global_path))
    print(global_path)
    # configure the logging path.
    conf.time_id = time2str()
    conf.logging_path = join(global_path, './logs', conf.time_id)
    conf.writting_path = join(conf.logging_path, './logging')
    # configure checkpoint for models.
    conf.model_directory = join(conf.logging_path, conf.MODEL_SAVEING_DIRECTORY)
    build_dirs(conf.model_directory)
    build_dirs(conf.writting_path)
    conf.writer = tensorboardX.SummaryWriter(log_dir=conf.writting_path,
                                            comment='video_sentiment_' + conf.time_id)


    # Setting parameters
    conf.max_epochs = args.epochs
    print('number epochs: {}'.format(conf.max_epochs))
    conf.num_data_workers = args.workers
    print('number of workers: {}'.format(conf.num_data_workers))
    conf.lr = args.learning_rate
    print('learning rate: {}'.format(conf.lr))
    conf.batch_size = args.batch_size
    print('batch size: {}'.format(conf.batch_size))
    conf.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Running engine: {}'.format(conf.device))
    conf.num_feature = args.num_feature
    conf.num_class = args.num_class
    conf.input_size = args.input_size
    conf.feature_extract = args.feature_extract

    train(conf)

def train(conf):
    # Load training and testing dataloader
    train_loader, test_loader = loading_data(conf)
    # Get initial model from initial function
    model_ft = init_model(conf)
    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if conf.feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=conf.lr, weight_decay=0.0005, momentum=0.9)
    scheduler_ft = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)

    # Setup the loss function
    criterion = nn.CrossEntropyLoss()

    # Set global iteration start as 0
    conf.iteration = 0
    conf.best_score = 0

    for epoch in range(conf.NUM_EPOCHS):
        # set internal iteration as 0
        conf.epoch_it = 0
        print('Epoch {}/{}'.format(epoch, conf.NUM_EPOCHS - 1))
        print('-' * 10)

        train_loss = 0.0
        senti_corr = 0
        cls_corr = 0
        block_length = 0
        # Begin iterate the dataloader for training process
        for inputs, labels in train_loader:
            # Set model training
            model_ft.train()
            # Iteration add up
            conf.iteration += 1
            conf.epoch_it += 1
            # Upload training data to cloud for training process
            inputs = inputs.to(conf.device)
            labels = labels.to(conf.device)

            # zero the parameter gradients
            optimizer_ft.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                # Get model outputs and calculate loss
                # Special case for inception because in training it has an auxiliary output. In train
                #   mode we calculate the loss by summing the final output and the auxiliary output
                #   but in testing we only consider the final output.
                sentimap_output, cls_output = model_ft(inputs)
                senti_loss = criterion(sentimap_output,labels)
                cls_loss = criterion(cls_output, labels)
                loss = 0.5 * senti_loss + 0.5 * cls_loss
                # Get result for evaluation
                _, senti_preds = torch.max(sentimap_output, 1)
                _, cls_preds = torch.max(cls_output, 1)
                # Statistic
                train_loss += loss.item() * inputs.size(0)
                senti_corr += torch.sum(senti_preds == labels.data)
                cls_corr += torch.sum(cls_preds == labels.data)
                # Generating training block length for calculating the average loss,acc,etc..
                block_length += inputs.size(0)
                # Backward begin
                loss.backward()
                optimizer_ft.step()
                scheduler_ft.step()
                conf.writer.add_scalar('WSCNet_sentiloss', senti_loss.data.cpu().numpy(),
                                       global_step=conf.iteration)
                conf.writer.add_scalar('WSCNet_clsloss', cls_loss.data.cpu().numpy(),
                                       global_step=conf.iteration)
            # Calculate average loss, acc
            train_loss = train_loss / block_length
            avg_senti_acc = senti_corr.double() / block_length
            avg_cls_acc = cls_corr.double() / block_length
            print('Loss: {:.4f} senti: {:.4f} cls: {:.4f}'.format( train_loss, avg_senti_acc, avg_cls_acc))
            conf.writer.add_scalar('WSCNet_train_senti_acc', avg_senti_acc,
                                   global_step=conf.iteration)
            conf.writer.add_scalar('WSCNet_train_cls_acc', avg_cls_acc,
                                   global_step=conf.iteration)

        # Every 30 iteration check the test score
        val_acc_senti, val_acc_cls = test_model(conf, model_ft, test_loader)
        # Print accuracy every 30 iterations
        print("Test senti: %2.3f ,cls: %2.3f" % (val_acc_senti, val_acc_cls))
        conf.writer.add_scalar('WSCNet_val_senti', val_acc_senti,
                               global_step=conf.iteration)
        conf.writer.add_scalar('WSCNet_val_cls', val_acc_cls,
                               global_step=conf.iteration)


def test_model(conf, model_ft, test_loader):
    model_ft.eval()
    # Define test correct count
    test_senti_correct = 0
    test_cls_correct = 0
    # Loading test data loader and do inference iteratively
    for inputs, labels in test_loader:
        # Upload to GPU for inference
        inputs = inputs.to(conf.device)
        labels = labels.to(conf.device)
        # Get forward result
        sentimap_output, cls_output = model_ft(inputs)
        # Get result with max probability
        _, senti_preds = torch.max(sentimap_output, 1)
        _, cls_preds = torch.max(cls_output, 1)
        # Count add up
        test_senti_correct += torch.sum(senti_preds == labels.data)
        test_cls_correct += torch.sum(cls_preds == labels.data)
    # Calculate the average after the loop
    val_acc_senti = test_senti_correct.double() / conf.test_data_length
    val_acc_cls = test_cls_correct.double() / conf.test_data_length
    # Compare the new score to the old one, get best result model
    if val_acc_cls > conf.best_score:
        conf.best_score = val_acc_cls
        # Define saved model name
        model_filename = '%s/video_sentiment_%s.pkl' % (conf.model_directory, 'best')
        # Save the model
        torch.save(model_ft.state_dict(), model_filename)

    return val_acc_senti, val_acc_cls


def init_model(conf):
    # Initialize these variables which will be set in this if statement. Each of these
    # variables is model specific.
    model_ft = models.resnet101(pretrained=True)
    # Set the pretrained net to finetune or not
    set_parameter_requires_grad(model_ft, conf.FINTUNE_FC)
    # Gnerating Pooling layer for model
    # First pooling layer (pooling each map to a certain feature vector)
    spatial_pooling = nn.Sequential()
    spatial_pooling.add_module('class_wise', ClassWisePool(conf.num_feature))
    # Second pooling layer (pooling the vector to class number vector for
    # classification loss calculation)
    spatial_pooling_2 = nn.Sequential()
    spatial_pooling_2.add_module('class_wise', ClassWisePool(conf.num_class))
    # Add this self defined layer to the pretrain model
    model_ft = ResNetWSL(model_ft, conf.num_class, conf.num_feature,
                         spatial_pooling, spatial_pooling_2)
    # Upload model to the gpu/cpu for training
    model_ft = model_ft.to(conf.device)

    return model_ft


def loading_data(conf):
    # Define data transformation format
    data_transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(conf.input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    data_transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(conf.input_size),
        transforms.CenterCrop(conf.input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Get dataset from dataset file
    train_data = Dataset(conf.DATA_ROOT, train=True, split=1, transform=data_transform_train)
    test_data = Dataset(conf.DATA_ROOT, train=False, split=1, transform=data_transform_test)

    # Create dataloader for training and testing use
    train_loader = torch.utils.data.DataLoader(dataset=train_data,batch_size=conf.batch_size,
                                               shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=conf.batch_size,
                                               shuffle=True, num_workers=8)
    # Get length for future use
    conf.train_data_length = len(train_data)
    conf.test_data_length = len(test_data)

    return train_loader, test_loader

if __name__ == '__main__':
    main()



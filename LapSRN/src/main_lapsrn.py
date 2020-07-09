import torch
from architecture import Net, L1_Charbonnier_loss
from lap_dataset import DatasetFromFolder
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import get_training_set, get_test_set
import argparse
from os.path import exists, join, basename
from os import makedirs, remove

parser = argparse.ArgumentParser(description='PyTorch LapSRN')
parser.add_argument('--batchSize', type=int, default=64, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=10, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--checkpoint', type=str, default='./model', help='Path to checkpoint')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning Rate. Default=0.0001')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
opt = parser.parse_args()

def main():

    global opt, model
    opt = parser.parse_args()
    print(opt)

    cudnn.benchmark = True

    print('Loading datasets')
    train_set = get_training_set()
    test_set = get_test_set()
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

    model = Net()
    criterion = L1_Charbonnier_loss()

    print("GPU will be used if found")
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("no checkpoint found at '{}'".format(opt.resume))

    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            model.load_state_dict(weights['model'].state_dict())
        else:
            print("no model found at '{}'".format(opt.pretrained)) 

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    print("Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1): 
        train(training_data_loader, optimizer, model, criterion, epoch)
        test()
        if epoch%100:
            save_checkpoint(model, epoch)

def adjust_learning_rate(optimizer, epoch):
    """Halves the learning rate to the initial LR by 10 every 100 epochs"""
    lr = opt.lr * (0.5 ** (epoch // 100))
    return lr

def train(training_data_loader, optimizer, model, criterion, epoch):

    lr = adjust_learning_rate(optimizer, epoch-1)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))

    model.train()

    for iteration, batch in enumerate(training_data_loader, 1):

        input, label_x2, label_x4 = Variable(batch[0]), Variable(batch[1], requires_grad=False), Variable(batch[2], requires_grad=False)

        if torch.cuda.is_available():
            input = input.cuda()
            label_x2 = label_x2.cuda()
            label_x4 = label_x4.cuda()

        HR_2x, HR_4x = model(input)

        loss_x2 = criterion(HR_2x, label_x2)
        loss_x4 = criterion(HR_4x, label_x4)
        loss = loss_x2 + loss_x4

        optimizer.zero_grad()

        loss_x2.backward(retain_graph=True)

        loss_x4.backward()

        optimizer.step()

        if iteration%100 == 0:
            print("Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(training_data_loader), loss.data[0]))

def save_checkpoint(model, epoch):
    model_folder = "checkpoint/"
    model_out_path = model_folder + "lapsrn_model_epoch_{}.pth".format(epoch)
    state = {"epoch": epoch ,"model": model, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))

def test():
    avg_psnr1 = 0
    avg_psnr2 = 0
    for batch in testing_data_loader:
        LR, HR_2_target, HR_4_target = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])
        
        if torch.cuda.is_available():
            LR = LR.cuda()
            HR_2_target = HR_2_target.cuda()
            HR_4_target = HR_4_target.cuda()

        HR_2, HR_4 = model(LR)
        mseloss = nn.MSELoss()
        mse1 = mseloss(HR_2, HR_2_target)
        mse2 = mseloss(HR_4, HR_4_target)
        psnr1 = 10 * log10(1 / torch.Tensor.tolist(mse1))
        psnr2 = 10 * log10(1 / torch.Tensor.tolist(mse2))
        avg_psnr1 += psnr1
        avg_psnr2 += psnr2
    print("Avg. PSNR for 2x: {:.4f} dB".format(avg_psnr1 / len(testing_data_loader)))
    print("Avg. PSNR for 4x: {:.4f} dB".format(avg_psnr2 / len(testing_data_loader)))

if __name__ == "__main__":
    main()
import argparse

# globel param
# dataset setting
img_width = 256
img_height = 128
img_channel = 3
label_width = 256
label_height = 128
label_channel = 1
data_loader_numworkers = 8
class_num = 2

# path
train_path = "/content/gdrive/MyDrive/Colab Notebooks/CNN cina/LaneDetectionCode/data/traningset_duck/indexLabels.txt"
# /content/gdrive/MyDrive/Colab Notebooks/CNN cina/LaneDetectionCode/data/train_index.txt
# /content/gdrive/MyDrive/Colab Notebooks/CNN cina/LaneDetectionCode/data/traningset_duck/indexLabels.txt
val_path = "/content/gdrive/MyDrive/Colab Notebooks/CNN cina/LaneDetectionCode/data/traningset_duck/indexLabels.txt"
#/content/gdrive/MyDrive/Colab Notebooks/CNN cina/LaneDetectionCode/data/val_index.txt
test_path = "/content/gdrive/MyDrive/Colab Notebooks/CNN cina/LaneDetectionCode/data/traningset_duck/TestDuck_index.txt"
save_path = "/content/gdrive/MyDrive/Colab Notebooks/CNN cina/LaneDetectionCode/save/result"
pretrained_path='/content/gdrive/MyDrive/Colab Notebooks/CNN cina/LaneDetectionCode/pretrained/16epoch.pth'
#/content/gdrive/MyDrive/Colab Notebooks/CNN cina/LaneDetectionCode/data/test_index_demo.txt
# weight
class_weight = [0.02, 1.02]

def args_setting():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch UNet-ConvLSTM')
    parser.add_argument('--model',type=str, default='UNet-ConvLSTM',help='( UNet-ConvLSTM | SegNet-ConvLSTM | UNet | SegNet | ')
    parser.add_argument('--batch-size', type=int, default=15, metavar='N',
                        help='input batch size for training (default: 10)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='use CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    return args

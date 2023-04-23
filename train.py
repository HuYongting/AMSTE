import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import time
from dataloader import DataLoader
from utils import *
from loss import *
from evaluate import val
from torch.autograd import Variable

import argparse

parser = argparse.ArgumentParser(description="MNAD")
parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for test')
parser.add_argument('--epochs', type=int, default=60, help='number of epochs for training')
parser.add_argument('--print_iter', type=int, default=200, help='number of epochs for training')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--c', type=int, default=3, help='channel of input images')
parser.add_argument('--method', type=str, default='pred', help='The target task for anoamly detection')
parser.add_argument('--t_length', type=int, default=5, help='length of the frame sequences')
parser.add_argument('--num_workers', type=int, default=2, help='number of workers for the train loader')
parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
parser.add_argument('--dataset_path', type=str, default='/home/huyt/DATASET', help='directory of data')
parser.add_argument('--RESUME', type=bool, default=False, help='directory of log')

parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
parser.add_argument('--dataset_type', type=str, default='ped2', help='type of dataset: ped2, avenue, shanghai')
parser.add_argument('--version', type=str, default='SAM_DCN_FUSION', help="oral:?????")
parser.add_argument('--lr', type=float, default=5e-5, help='initial learning rate')
parser.add_argument('--para', type=str, default="{p,gamma=0.0,30_512}", help='参数')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

set_seed(114514)   # 42,3407,114514
#下面这两句设置成False可以保证结果可复现，都设成True可以提升效率，输入变化不大时候全部设成True也可以复现
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# Loading dataset
if args.dataset_type == 'shanghai':
    train_folder = args.dataset_path + "/" + args.dataset_type + "/Train"
    args.epochs = 10
    args.print_iter = 10000
else:
    train_folder = args.dataset_path + "/" + args.dataset_type + "/training/frames"
    args.epochs = 60
    if args.dataset_type == 'avenue':
        args.print_iter = 1000
    else:
        args.print_iter = 200

train_dataset = DataLoader(train_folder, transforms.Compose([transforms.ToTensor(),]), resize_height=args.h, resize_width=args.w, time_step=args.t_length - 1)
train_batch = data.DataLoader(train_dataset, batch_size=args.batch_size,shuffle=False, num_workers=args.num_workers, drop_last=True)

''' ====================================   model   ======================================'''
from models.model import Model
model = Model(n_channel=3, t_length=5,memory_size=30,memory_dim=512).cuda()
params = list(model.parameters())
optimizer = torch.optim.Adam(params, lr=args.lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

''' ====================================  loss ========================================='''
tr_entropy_loss_func = EntropyLossEncap().cuda()
adversarial_loss = Adversarial_Loss().cuda()
discriminate_loss = Discriminate_Loss().cuda()
gradient_loss = Gradient_Loss(3).cuda()
flow_loss = Flow_Loss().cuda()
intensity_loss = Intensity_Loss().cuda()
triplet_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2)
intensity_5frame_loss = Intensity_5frame_Loss().cuda()
loss_func_mse = nn.MSELoss().cuda()

''' ====================================  manager ======================================='''
root_path = './results/' + args.dataset_type + '/' + args.version + '/'
makedir(root_path)
time_ = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
model_str = 'model_{'  + str(args.batch_size) + '}_'+ str(args.lr) + args.para
model_path = os.path.join(root_path, model_str)
makedir(model_path)
logger = get_logger(os.path.join(model_path, time_ + "_exp.log"))  # ??.log??
for k in list(vars(args).keys()):
    logger.info('%s: %s' % (k, vars(args)[k]))
logger.info('--------------------------------------start training!-------------------------------------')

start_epoch = -1
for epoch in range(start_epoch + 1, args.epochs):
    model.train()
    for j, (data) in enumerate(train_batch):
        imgs = Variable(data[0]).cuda()               # b,15,256,256
        flows = Variable(data[1]).cuda().permute(0,3,1,2)  # b,256,256,10 -> b,10,256,256
        frame_p = model.forward(imgs[:,:12,:],flows[:,2:,:])
        loss_p = torch.mean(loss_func_mse(frame_p, imgs[:,12:]))
        loss = loss_p
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ((j + 1) % args.print_iter == 0):  # shanghai:10000    avenue: 1000   ped2: 200
            logger.info( '[{:04d} / {:04d}]  loss: {:.8f} '.format(epoch + 1, j + 1,loss.item()))
    scheduler.step()

    if ((epoch + 1) % args.epochs == 0):
        torch.save(model, os.path.join(model_path, 'model_{}.pth'.format(epoch + 1)))








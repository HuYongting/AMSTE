import time
import torch.utils.data as data
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
from dataloader import DataLoader

from sklearn.metrics import roc_auc_score
from utils import *
import random
import glob

import argparse

def val(args,model=None):
    if args.dataset_type == 'shanghai':
        test_folder = args.dataset_path + "/" + args.dataset_type + "/Test"
    else:
        test_folder = args.dataset_path + "/" + args.dataset_type + "/testing/frames"

    test_dataset = DataLoader(test_folder, transforms.Compose([transforms.ToTensor(),]), resize_height=args.h, resize_width=args.w, time_step=args.t_length - 1)
    test_batch = data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False,
                                 num_workers=args.num_workers_test, drop_last=False)
    labels = np.load(args.dataset_path + '/label/frame_labels_' + args.dataset_type + '.npy')

    if model:
        model = model.cuda()
        model.eval()
    else:
        model = torch.load(args.model_dir)
        model.cuda().eval()

    videos = OrderedDict()
    videos_list = sorted(glob.glob(os.path.join(test_folder, '*')))
    for video in videos_list:
        video_name = video.split('/')[-1]
        videos[video_name] = {}
        videos[video_name]['path'] = video
        videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
        videos[video_name]['frame'].sort()
        videos[video_name]['length'] = len(videos[video_name]['frame'])

    labels_list = []
    label_length = 0
    psnr_multi = {}

    # Setting for video anomaly detection
    for video in sorted(videos_list):
        video_name = video.split('/')[-1]
        if args.dataset_type == 'shanghai':
            labels_list = np.append(labels_list,labels[label_length + 4:videos[video_name]['length'] + label_length])       # shanghai
        else:
            labels_list = np.append(labels_list,labels[0][4 + label_length:videos[video_name]['length'] + label_length])  # ped2 and avenue
        label_length += videos[video_name]['length']
        psnr_multi[video_name] = []
    label_length = 0
    video_num = 0
    label_length += videos[videos_list[video_num].split('/')[-1]]['length']
    model.eval()
    for k, (datas) in enumerate(test_batch):
        if k * args.test_batch_size == label_length - 4 * (video_num + 1):
            video_num += 1
            label_length += videos[videos_list[video_num].split('/')[-1]]['length']
        imgs = Variable(datas[0]).cuda()
        flows = Variable(datas[1]).cuda().permute(0, 3, 1, 2)  # b,256,256,10 -> b,10,256,256
        outputs = model.forward(imgs[:, :12], flows[:, 2:, :])
        mse_pixel = (((outputs + 1) / 2) - ((imgs[:, 4 * args.c:, :, :] + 1) / 2)) ** 2
        mse_32, mse_64, mse_128 = multi_patch_max_mse(mse_pixel.cpu().detach().numpy())
        mse_multi = mse_32 + mse_64 + mse_128
        psnr_multi[videos_list[video_num].split('/')[-1]].append(psnr(mse_multi))
    psnr_multi_list = []
    for video in sorted(videos_list):
        video_name = video.split('/')[-1]
        psnr_multi_list.extend(multi_future_frames_to_scores(np.array(normalize_score_list_gel(psnr_multi[video_name]))))
    psnr_multi_list = np.asarray(psnr_multi_list)
    accuracy_multi = AUC(psnr_multi_list, np.expand_dims(1 - labels_list, 0))

    return accuracy_multi * 100

if __name__ == '__main__':
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

    model = torch.load("/home/huyt/ONE/weight/ped2.pth")
    s = val(args, model=model)
    print(s)

    # path = '/home/huyt/IdeaTest/results/avenue/contrast_model'
    # files = os.listdir(path)  # ¶ÁÈëÎÄ¼þ¼Ð
    #
    # for file in files:
    #     folder = os.path.join(path,file)
    #     model = torch.load(folder)
    #     epoch = folder.split("/")[-1].split(".")[0]
    #     _,s = val(args, model=model,epoch=epoch)
    #     print("{}:{}".format(epoch,s))







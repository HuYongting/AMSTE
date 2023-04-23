from fusion import *
import torch
import torch.nn as nn
from torch.nn import functional as F
try:
    from models.DCNv2.dcn_v2 import DCN_sep
except ImportError:
    raise ImportError('Failed to import DCNv2 module.')

# 第一种
class Trans(nn.Module):
    def __init__(self, nf=64):
        super(Trans, self).__init__()
        self.conv1_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(nf, nf, kernel_size=3, stride=2, padding=1, dilation=1)
        self.conv2_2 = nn.Conv2d(nf, nf, kernel_size=3, stride=2, padding=1, dilation=1)
        self.conv2_3 = nn.Conv2d(nf, nf, kernel_size=3, stride=2, padding=1, dilation=1)

        self.conv3_1 = nn.Conv2d(nf, nf, kernel_size=5, stride=2, padding=2, dilation=1)
        self.conv3_2 = nn.Conv2d(nf, nf, kernel_size=5, stride=2, padding=2, dilation=1)
        self.conv3_3 = nn.Conv2d(nf, nf, kernel_size=5, stride=2, padding=2, dilation=1)

        self.fusion = nn.Conv2d( 3 * nf, nf, kernel_size=1, stride=1, padding=0)
        self.linear = nn.Linear(64 * 64 ,512)

        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.bn = nn.BatchNorm2d(nf)


    def forward(self, x):  # (12,64,8,8)
        x1 = self.act(self.bn(self.conv1_1(x)))
        x1 = self.act(self.bn(self.conv1_2(x1)))
        x1 = self.act(self.bn(self.conv1_3(x1)))

        x2 = self.act(self.bn(self.conv2_1(x)))
        x2 = self.act(self.bn(self.conv2_2(x2)))
        x2 = self.act(self.bn(self.conv2_3(x2)))

        x3 = self.act(self.bn(self.conv3_1(x)))
        x3 = self.act(self.bn(self.conv3_2(x3)))
        x3 = self.act(self.bn(self.conv3_3(x3)))

        x_fusion = torch.cat([x1,x2,x3],dim=1)
        out = self.act(self.fusion(x_fusion))
        out = self.linear(out.squeeze(0).flatten(1,2))
        return out

# 第二种
class Trans2(nn.Module):
    def __init__(self, nf=64):
        super(Trans2, self).__init__()
        self.conv1_1 = nn.Conv2d(nf, nf, kernel_size=5, stride=2, padding=1)
        self.conv1_2 = nn.Conv2d(nf, nf, kernel_size=3, stride=2, padding=1)
        self.trans = nn.Linear(32*32,512)

    def forward(self, x):    #   (12,64,8,8)
        x1 =self.conv1_1(x)
        x1 = self.conv1_1(x1)
        x1 = self.conv1_2(x1)
        x1 = self.conv1_2(x1)
        x1 = x1.squeeze(0)
        x1 = x1.flatten(1,2)
        x1 = self.trans(x1)

        return x1

# 外积
def op_att(q, k, v):
    qq = q.unsqueeze(2).repeat(1, 1, k.shape[1], 1)
    kk = k.unsqueeze(1).repeat(1, q.shape[1], 1, 1)
    output = torch.matmul(torch.tanh(qq*kk).unsqueeze(4), v.unsqueeze(1).repeat(1, q.shape[1], 1, 1).unsqueeze(3))  # BxNXNxd_kq BxNxNxd_v --> BxNXNxd_kqxd_v
    # print(output.shape)
    output = torch.sum(output, dim=2)  # BxNxd_kqxd_v
    # print(output.shape)
    return output

class Encoder(torch.nn.Module):
    def __init__(self, t_length=5, n_channel=3):
        super(Encoder, self).__init__()

        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )

        def Basic_(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
            )

        self.moduleConv1 = Basic(n_channel * (t_length - 1), 64)
        self.modulePool1 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

        self.moduleConv2 = Basic(64, 128)
        self.modulePool2 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)

        self.moduleConv3 = Basic(128, 256)
        self.modulePool3 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)

        self.moduleConv4 = Basic_(256, 512)
        self.moduleBatchNorm = torch.nn.BatchNorm2d(512)
        self.moduleReLU = torch.nn.ReLU(inplace=False)

    def forward(self, x):
        tensorConv1 = self.moduleConv1(x)
        tensorPool1 = self.modulePool1(tensorConv1)

        tensorConv2 = self.moduleConv2(tensorPool1)
        tensorPool2 = self.modulePool2(tensorConv2)

        tensorConv3 = self.moduleConv3(tensorPool2)
        tensorPool3 = self.modulePool3(tensorConv3)

        tensorConv4 = self.moduleConv4(tensorPool3)

        return tensorConv4, tensorConv1, tensorConv2, tensorConv3

class Decoder(torch.nn.Module):
    def __init__(self, t_length=5, n_channel=3):
        super(Decoder, self).__init__()

        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )

        def Gen(intInput, intOutput, nc):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.Tanh()
            )

        def Upsample(nc, intOutput):
            return torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=nc, out_channels=intOutput, kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )


        self.moduleConv = Basic(512 * 2, 512)
        self.moduleUpsample4 = Upsample(512, 256)

        self.moduleDeconv3 = Basic(512, 256)
        self.moduleUpsample3 = Upsample(256, 128)

        self.moduleDeconv2 = Basic(256, 128)
        self.moduleUpsample2 = Upsample(128, 64)

        self.moduleDeconv1 = Gen(128, n_channel, 64)

        self.dcnpack3 = DCN_sep(256, 256, 3, stride=1, padding=1, dilation=1, deformable_groups=8)
        self.dcnpack2 = DCN_sep(128, 128, 3, stride=1, padding=1, dilation=1, deformable_groups=8)
        self.lrelu = torch.nn.ReLU(inplace=False)

    def forward(self, x, skip1, skip2, skip3):

        tensorConv = self.moduleConv(x)                   # b,512,32,32

        tensorUpsample4 = self.moduleUpsample4(tensorConv)  # b,256,64,64
        cat4 = torch.cat((skip3, tensorUpsample4), dim=1)
        tensorDeconv3 = self.moduleDeconv3(cat4)
        tensorDeconv3 = self.lrelu(self.dcnpack3(skip3,tensorDeconv3))

        tensorUpsample3 = self.moduleUpsample3(tensorDeconv3)
        cat3 = torch.cat((skip2, tensorUpsample3), dim=1)
        tensorDeconv2 = self.moduleDeconv2(cat3)
        tensorDeconv2 = self.lrelu(self.dcnpack2(skip2, tensorDeconv2))

        tensorUpsample2 = self.moduleUpsample2(tensorDeconv2)
        cat2 = torch.cat((skip1, tensorUpsample2), dim=1)
        output = self.moduleDeconv1(cat2)

        return output


class Memory(torch.nn.Module):
    def __init__(self,memory_size=100,memory_dim=512,init_alphas = [1.0,1.0,1.0]):
        super(Memory, self).__init__()
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.item_shape = [memory_dim, memory_dim]
        self.item_w = nn.init.normal_(torch.empty(self.item_shape), mean=0.0, std=1.0)
        self.item_w = nn.Parameter(self.item_w, requires_grad=True)
        self.rel_shape = [memory_size, memory_dim,memory_dim]
        self.rel_w = nn.init.normal_(torch.empty(self.rel_shape), mean=0.0, std=1.0)
        self.rel_w = nn.Parameter(self.rel_w, requires_grad=True)

        self.input_project1 = torch.nn.Linear(512,self.memory_dim,True)
        self.input_project2 = torch.nn.Linear(512,self.memory_dim,True)
        self.input_project3 = torch.nn.Linear(512,self.memory_size,True)
        self.qkv_project = torch.nn.Linear(self.memory_dim,self.memory_size * 3,True)
        self.qkv_layernorm = nn.LayerNorm(self.memory_size * 3 , self.memory_dim)
        self.rel_project1 = torch.nn.Linear(self.memory_dim * self.memory_dim,512,True)
        self.rel_project2 = torch.nn.Linear(self.memory_dim * self.memory_size,self.memory_dim,True)
        self.rel_project3 = torch.nn.Linear(self.memory_dim * self.memory_size,512,True)


        if init_alphas[0] is None:
            self.alpha1 = [nn.Parameter(torch.zeros(1))]
            for ia, a in enumerate(self.alpha1):
                setattr(self, 'alpha1' + str(ia), self.alpha1[ia])
        else:
            self.alpha1 = [init_alphas[0]]

        if init_alphas[1] is None:
            self.alpha2 = [nn.Parameter(torch.zeros(1))]
            for ia, a in enumerate(self.alpha2):
                setattr(self, 'alpha2' + str(ia), self.alpha2[ia])
        else:
            self.alpha2 = [init_alphas[1]]

        if init_alphas[2] is None:
            self.alpha3 = [nn.Parameter(torch.zeros(1))]
            for ia, a in enumerate(self.alpha3):
                setattr(self, 'alpha3' + str(ia), self.alpha3[ia])
        else:
            self.alpha3 = [init_alphas[2]]



    def forward(self,fea):   #  (b,512,32,32)
        # memory addressing
        b,c,h,w = fea.size()
        fea_reshape = fea.permute(0,2,3,1).reshape(b*h*w,c)
        query_norm = F.normalize(fea_reshape, dim=1)
        input_project1 = self.input_project1(query_norm)
        input_project2 = self.input_project2(query_norm)
        input_project3 = self.input_project3(query_norm)
        input_project3 = F.softmax(input_project3,dim=-1)
        input_project4 = torch.einsum('bn,bd,ndf->bf', input_project3, input_project2, self.rel_w)
        item0 = torch.einsum('bd,cf->df', input_project4, input_project2)
        item1 = self.item_w + torch.matmul(input_project1.t(),input_project1)
        qkv_project = self.qkv_project(item0.unsqueeze(0) + self.alpha2[0] * item1.unsqueeze(0)).squeeze(0)
        qkv_project = self.qkv_layernorm(qkv_project)
        q,k,v = torch.split(qkv_project,[self.memory_size ]* 3,1)
        rel0 = op_att(q.unsqueeze(0).permute(0,2,1),k.unsqueeze(0).permute(0,2,1),v.unsqueeze(0).permute(0,2,1))
        item2 = self.rel_project2(rel0.view(rel0.shape[0], -1, rel0.shape[3]).permute(0, 2, 1))
        self.item_w = nn.Parameter(item1 + self.alpha3[0] * item2.squeeze(0))
        self.rel_w = nn.Parameter(( self.rel_w + self.alpha1[0] * rel0).squeeze(0))

        memory = self.rel_project1(self.rel_w.view(self.memory_size,-1))  # (30,512)
        memory_norm = F.normalize(memory, dim=1)
        s = torch.mm(query_norm, memory_norm.transpose(dim0=0, dim1=1))
        addressing_vec = F.softmax(s, dim=1)
        memory_feature = torch.mm(addressing_vec, memory_norm)

        memory_feature = memory_feature.reshape(b, h, w, c).permute(0, 3, 1, 2)
        updated_fea = torch.cat([fea, memory_feature], dim=1)
        # updated_fea = self.fusion(updated_fea)

        return updated_fea

def softmax_normalization(x, func):
    b, c, h, w = x.shape
    x_re = x.view([b, c, -1])
    x_norm = func(x_re)
    x_norm = x_norm.view([b, c, h, w])
    return x_norm

class Fusion(nn.Module):
    def __init__(self, in_channels):
        super(Fusion, self).__init__()
        self.convB = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False)
        self.convC = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False)
        self.convD = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False)
        self.gamma = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.gamma.data.fill_(0.0)   #  avenue:1   ped2:0
        self.softmax = nn.Softmax(dim=2)
        self.norm_func = nn.Softmax(-1)

    def forward(self, x,img):  # flow,img
        B = self.convB(x)
        C = self.convC(x)
        S = self.softmax(torch.matmul(B, C))
        flow_mean = torch.mean(S, dim=1, keepdim=True)
        variance_x = torch.sum(torch.pow(S - flow_mean, 2), dim=1, keepdim=True)
        variance_x = softmax_normalization(variance_x, self.norm_func)  # ¡¤?2?
        D = self.convD(img)
        E = torch.matmul(D, torch.exp(variance_x) * x)
        E = self.gamma * E + img
        return E

class Model(torch.nn.Module):
    def __init__(self, n_channel=3, t_length=5,memory_size=30,memory_dim=512):
        super(Model, self).__init__()
        self.encoder = Encoder(t_length, n_channel)
        self.encoder_flow = Encoder(t_length , n_channel -1) # n_channel-1
        self.decoder_frame = Decoder(t_length, n_channel)
        self.memory = Memory(memory_size=memory_size,memory_dim=memory_dim)  #
        self.fusion_fea = Fusion05(memory_dim)

    def forward(self, x,flow):
        fea_frame, skip1_frame, skip2_frame, skip3_frame = self.encoder(x)  # (b,512,32,32) (b,64,256,256) (b,128,128,128) (b,256,64,64)
        fea_flow, skip1_flow, skip2_flow, skip3_flow = self.encoder_flow(flow)  # (b,512,32,32) (b,64,256,256) (b,128,128,128) (b,256,64,64)
        fea = self.fusion_fea(fea_flow,fea_frame)
        fea = self.memory(fea)
        out_frame = self.decoder_frame(fea, skip1_frame, skip2_frame, skip3_frame)
        return out_frame # ,fea_frame,fea_flow


if __name__ == '__main__':
    imgs = torch.rand(2,12,256,256).cuda()
    model = Model(n_channel=3, t_length=5,memory_size=30,memory_dim=512).cuda()
    res = model.forward(imgs)
    print(res.shape)
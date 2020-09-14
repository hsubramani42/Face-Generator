import torch
from torch.nn import *
#image size (batch_size, input_channel , 128, 128)
class Discriminator(torch.nn.Module):
    def __init__(self,input_channel:int,first_out:int):
        super(Discriminator, self).__init__()
        out_channel=first_out
        self.discriminator=Sequential(
                         #(n,3,128,128)
                         Conv2d(input_channel,out_channel,kernel_size=(4,4),stride=(2,2),padding=1),
                         BatchNorm2d(out_channel),
                         LeakyReLU(0.2,inplace=True),

                         #(n,out_channel,32,32)
                         Conv2d(out_channel,out_channel*2,kernel_size=(4,4),stride=(2,2),padding=1),
                         BatchNorm2d(out_channel*2),
                         LeakyReLU(0.2,inplace=True),

                         #(n,out_channel*2,16,16)
                         Conv2d(out_channel*2,out_channel*4,kernel_size=(4,4),stride=(2,2),padding=1),
                         BatchNorm2d(out_channel*4),
                         LeakyReLU(0.2,inplace=True),

                         #(n,out_channel*4,8,8)
                         Conv2d(out_channel*4,out_channel*8,kernel_size=(4,4),stride=(2,2),padding=1),
                         BatchNorm2d(out_channel*8),
                         LeakyReLU(0.2,inplace=True),

                         #(n,out_channel*8,4,4)
                         Conv2d(out_channel*8,out_channel*16,kernel_size=(4,4),stride=(2,2),padding=1),
                         BatchNorm2d(out_channel*16),
                         LeakyReLU(0.2,inplace=True),

                         #(n,out_channel*16,2,2)
                         Conv2d(out_channel*16,out_channel*32,kernel_size=(4,4),stride=(2,2),padding=1),
                         BatchNorm2d(out_channel*32),
                         LeakyReLU(0.2,inplace=True),

                         #(n,out_channel*32,1,1)
                         Conv2d(out_channel*32,out_channel*64,kernel_size=(4,4),stride=(2,2),padding=1),
                         LeakyReLU(0.2,inplace=True)
                         Flatten(),
                         
                         #(n,out_channel*64)
                         Linear(out_channel*64,512),
                         BatchNorm1d(512),
                         LeakyReLU(0.2,inplace=True),

                         #(n,512)
                         Linear(512,1),

                         #(n,1)
                         Sigmoid()
                         )
    def forward(self,xb):
        return self.discriminator(xb.cuda())
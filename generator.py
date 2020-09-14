class Generator(torch.nn.Module):
    #image output size is (batch_size,out_channel,128,128)
    def __init__(self,input_channel,out_channel):
        super(Generator, self).__init__()
        self.generator=Sequential(
                    #(n,input_channel,1,1)
                    ConvTranspose2d(input_channel,512,kernel_size=(4,4),stride=(1,1),padding=0),
                    BatchNorm2d(512),
                    ReLU(),

                    #(n,512,4,4)
                    ConvTranspose2d(512,256,kernel_size=(4,4),stride=(2,2),padding=1),
                    BatchNorm2d(256),
                    ReLU(),

                    #(n,256,8,8)
                    ConvTranspose2d(256,128,kernel_size=(4,4),stride=(2,2),padding=1),
                    BatchNorm2d(128),
                    ReLU(),

                    #(n,128,16,16)
                    ConvTranspose2d(128,64,kernel_size=(4,4),stride=(2,2),padding=1),
                    BatchNorm2d(64),
                    ReLU(),
                    
                    #(n,64,32,32)
                    ConvTranspose2d(64,32,kernel_size=(4,4),stride=(2,2),padding=1),
                    BatchNorm2d(32),
                    ReLU(),

                    #(n,16,64,64)
                    ConvTranspose2d(32,3,kernel_size=(4,4),stride=(2,2),padding=1),
                    Tanh()
                    
                    #(n,3,128,128)
                    )
        
    def forward(self,xb):
        return self.generator(xb.cuda())
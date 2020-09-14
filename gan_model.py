import torch
import sys
import numpy as np
torch.manual_seed(0)
class GAN:
    def __init__(self,discriminator:torch.nn.Module,generator:torch.nn.Module,
                 train_dl:torch.utils.data.Dataset,latent_size:int,batch_size:int):
        torch.manual_seed(0) #reproducible outputs
        self.discriminator=discriminator.cuda().float()
        self.batch_size=batch_size
        self.generator=generator.cuda().float()
        self.train_dl=train_dl
        self.tracker={'dis':[],'gen':[]}
        self.latent=torch.randn(batch_size,latent_size,1,1).cuda()
        self.loss=torch.nn.BCELoss()
        self.dis_optim=torch.optim.Adam(self.discriminator.parameters(),lr=0.0002,betas=(0.5, 0.999))
        self.gen_optim=torch.optim.Adam(self.generator.parameters(),lr=0.0002,betas=(0.5, 0.999))
        self.latest=None
    def train_discriminator(self,xb:torch.tensor):
        self.dis_optim.zero_grad()
        #passing real images into discriminator and finding loss of the model
        out_real=self.discriminator(xb.cuda().float())
        loss_real=self.loss(out_real,torch.ones(out_real.shape[0],1).cuda())
        #generate a fake image batch
        fake_img=self.generator(self.latent)
        #passing fake image into discriminator and finding loss of the model
        out_fake=self.discriminator(fake_img)
        loss_fake=self.loss(out_fake,torch.zeros(out_fake.shape[0],1).cuda())
        #parameters update
        dis_loss=loss_fake+loss_real
        dis_loss.backward()
        #return loss of the model
        self.dis_optim.step()
        return dis_loss.item()
    def train_generator(self,xb:torch.tensor):
        self.gen_optim.zero_grad()
        #generate fake images
        fake_img=self.generator(self.latent)
        self.latest=fake_img.detach().cpu()
        #passing images to discriminator to fool it
        out_fake=self.discriminator(fake_img)
        #update generator
        gen_loss=self.loss(out_fake,torch.ones(out_fake.shape[0],1).cuda())
        gen_loss.backward()
        self.gen_optim.step()
        #return loss of the model
        return gen_loss.item()
    def img_generator(self):
        """to display a random generated images"""
        idx=np.random.randint(self.batch_size)
        outs=self.latest
        img=outs[idx].permute(1,2,0).numpy()
        img=img * 0.5 +0.5
        plt.imshow(img)
        plt.show()
    def loss_tracker(self):
        """to plot loss of the models w.r.t """
        import matplotlib.pyplot as plt
        plt.plot(np.arange(0,len(self.tracker['dis'])),self.tracker['dis'],
                 'blue',label='Discriminator Loss')
        plt.plot(np.arange(0,len(self.tracker['gen'])),self.tracker['gen'],
                 'orange',label='Generator Loss')
        plt.show()
    def show_batch(self,nrow:int,epoch:int):
        import matplotlib.pyplot as plt
        from torchvision.utils import save_image
        #changing the tanh distribution to interval [0,1]
        img_batch=self.latest * 0.5 + 0.5
        #save_image in current working directory
        save_image(img_batch,fp='Generated at epoch'+str(epoch)+'.png',nrow=nrow,scale_each=True)
    def fit(self,epochs):
        import matplotlib.pyplot as plt
        for i in range(epochs):
            dis_loss,gen_loss=0,0
            fake_image=None
            for j,(xb) in enumerate(self.train_dl):
                dis_loss=self.train_discriminator(xb)
                gen_loss=self.train_generator(xb)
                self.tracker['dis'].append(dis_loss)
                self.tracker['gen'].append(gen_loss)
                #display running progress
                out=('No.of batches passed in training: '+str(j)+' *&*  Progress Percentage: '+str(np.round(j*100/len(self.train_dl),decimals=2))+'%')
                sys.stdout.write('\r'+out)
            #clear progress
            sys.stdout.write('\r')    
            dis_loss=np.round(self.tracker['dis'][-1],decimals=4)
            gen_loss=np.round(self.tracker['gen'][-1],decimals=4)
            #print loss of the models
            print(f'Epoch: {i+1} *&* Discriminator loss: {dis_loss} *&* Generator loss: {gen_loss}')
            self.show_batch(int(self.batch_size/8),i)
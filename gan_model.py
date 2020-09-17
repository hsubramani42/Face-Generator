import matplotlib.pyplot as plt
from torchvision.utils import save_image,make_grid
import sys
import numpy as np
import torch
torch.manual_seed(999)
class GAN:
    def __init__(self,discriminator:torch.nn.Module,generator:torch.nn.Module,
                 train_dl:torch.utils.data.Dataset,latent_size:int,batch_size:int):
        self.discriminator=discriminator.cuda().float()
        self.batch_size=batch_size
        self.generator=generator.cuda().float()
        self.train_dl=train_dl
        self.tracker={'dis':[],'gen':[]}
        self.latent=torch.randn(batch_size,latent_size,1,1).cuda()
        self.loss=torch.nn.BCELoss()
        self.dis_optim=None
        self.gen_optim=None
        self.latest=None
    def train_discriminator(self,xb:torch.tensor):
        self.dis_optim.zero_grad()
        #passing real images into discriminator and finding loss of the model
        out_real=self.discriminator(xb.cuda().float())
        loss_real=self.loss(out_real,torch.ones(out_real.shape[0],1).cuda())
        #generate a fake image batch
        fake_img=self.generator(self.latent)
        self.latest=fake_img
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
        plt.plot(np.arange(0,len(self.tracker['dis'])),self.tracker['dis'],
                 'blue',label='Discriminator Loss')
        plt.plot(np.arange(0,len(self.tracker['gen'])),self.tracker['gen'],
                 'orange',label='Generator Loss')
        plt.show()
    def show_batch(self,nrow:int,epoch:int):
        noise=self.latest * 0.5 + 0.5
        save_image(noise,fp='Generated at epoch'+str(epoch)+'.png',nrow=nrow,scale_each=True)
        img=np.transpose(make_grid(noise).numpy(),(1,2,0))
        plt.imshow(img)
        plt.show()
    def best_image_generator(self,nrow):
        self.generator.load_state_dict(torch.load("best_generator_parameters.pth"))
        self.latest=self.generator(self.latent.cuda()).detach().cpu()
        noise=self.latest * 0.5 + 0.5
        save_image(noise,fp='Best Generated.png',nrow=nrow,scale_each=True)
        img=np.transpose(grid(noise),(1,2,0))
        plt.imshow(img)
        plt.show()
        self.latest
    def model_save(self,d_loss,g_loss):
        #comparing model with previous states and saving best parameters
        if(len(self.tracker['dis'])>0):
            if((self.tracker['dis'][-1] < d_loss) & (self.tracker['gen'][-1] < g_loss)):
                return True
        return False
    def fit(self,epochs,dis_lr=0.002,gen_lr=0.002):
        self.dis_optim=torch.optim.Adam(self.discriminator.parameters(),lr=dis_lr,betas=(0.5, 0.999))
        self.gen_optim=torch.optim.Adam(self.generator.parameters(),lr=gen_lr,betas=(0.5, 0.999))
        for i in range(epochs):
            dis_epoch_loss,gen_epoch_loss=0,0
            fake_image=None
            for j,(xb) in enumerate(self.train_dl):
                dis_loss=self.train_discriminator(xb)
                gen_loss=self.train_generator(xb)
                #check loss with previous loss and store best weights of model
                if(self.model_save(dis_loss,gen_loss)):
                    torch.save(self.generator.state_dict(),'best_generator_parameters.pth')
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
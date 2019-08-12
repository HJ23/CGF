import torch
import torchvision
from input_loader import initialize_dataloader
from config import configs
import network
from matplotlib import pyplot
import cv2
import numpy

class train_starter():
    def __init__(self):
        self.dataloader=initialize_dataloader()

        self.generator=network.generator().to(configs.device)
        self.discriminator=network.discriminator().to(configs.device)

        self.optimizer_generator=torch.optim.Adam(self.generator.parameters(),lr=configs.learning_rate,betas=configs.betas)
        self.optimizer_discriminator=torch.optim.Adam(self.discriminator.parameters(),lr=configs.learning_rate,betas=configs.betas)
        
        self.loss=torch.nn.BCELoss()
        self.disloss=[]
        self.genloss=[]
    def plot(self):
        pyplot.plot(self.disloss,label="discriminator loss")
        pyplot.plot(self.genloss,lable="generator loss")
        pyplot.show()

    def generate(self,input):        
        return self.generator(input)

    def load_pretrained(self,pathgen,pathdis):
        self.generator.load_state_dict(torch.load(pathgen))
        self.discriminator.load_state_dict(torch.load(pathdis))
        self.generator.eval()
        self.discriminator.eval()

    def tensor2image(self,image):
        imggg=image.permute(1,2,0).numpy()
        imggg=cv2.cvtColor(imggg,cv2.COLOR_BGR2GRAY)
        imggg=255*(imggg-numpy.mean(imggg))/(numpy.max(imggg)-numpy.min(imggg))
        imggg=imggg.astype(numpy.uint8)
        imggg=cv2.resize(imggg,(200,200))
        return imggg


    def save_models(self,epoch_count):
        torch.save(self.generator.state_dict(),"../pretrained/G"+str(epoch_count)+".pth")
        torch.save(self.discriminator.state_dict(),"../pretrained/D"+str(epoch_count)+".pth")
    
    def start(self):
        cv2.namedWindow("generated")
        for epoch in range(configs.epoch):
            print("*****************************************************")
            for i,batch in enumerate(self.dataloader,0):
                self.discriminator.zero_grad()
                real=torch.ones((batch[0].shape[0],),dtype=torch.float32,device=configs.device).view(-1)
                fake=torch.zeros((batch[0].shape[0],),dtype=torch.float32,device=configs.device).view(-1)
                out=self.discriminator(batch[0].to(configs.device)).view(-1)
                lossDreal=self.loss(out,real)
                self.disloss.append(lossDreal)
                lossDreal.backward()

                # noise for generator
                Z=torch.randn(batch[0].shape[0],100,1,1).to(configs.device)
                generated=self.generator(Z)
                out=self.discriminator(generated.detach()).view(-1)
                lossDfake=self.loss(out,fake)
                lossDfake.backward()
                lossD=(lossDfake+lossDreal)/2
                self.disloss.append(lossD)

                self.optimizer_discriminator.step()

                self.generator.zero_grad()
                out=self.discriminator(generated).view(-1)
                lossg=self.loss(out,real)
                lossg.backward()
                self.genloss.append(lossg)
                self.optimizer_generator.step()

                if(i%30==0):
                    imggg=self.tensor2image(generated[0].detach())
                    cv2.imshow("generated",imggg)
                    cv2.waitKey(1)
            if((epoch+1)%configs.save_pretrained_epochnum==0):
                self.save_models(epoch)


if __name__=='__main__':
    tr=train_starter()

    
    tr.start()
    

    # For pretrained model use this block commet code block above
    ##########################################################################
    #tr.load_pretrained("../pretrained/G5.pth","../pretrained/D5.pth")
    #for x in range(10):
    #    generated=tr.generate(torch.randn((1,100,1,1)))
    #    generated=tr.tensor2image(generated[0].detach())
    #    cv2.imshow("image",generated)
    #    cv2.waitKey(4000)

























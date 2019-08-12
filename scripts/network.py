import torch.nn as nn

class network(nn.Module):
    def __init__(self,object):
        super(network,object).__init__()
    
    def __repr__(self):
        print("This is just network base class")
    def  forward(self,input):
        pass


class generator(network):
    def __init__(self):
        network.__init__(self,self)
        self.stack=[]
        self.stack+=[nn.ConvTranspose2d( 100, 512, 4, 1, 0, bias=False)]
        self.stack+=[nn.BatchNorm2d(512)]
        self.stack+=[nn.ReLU(True)]
        self.stack+=[nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False)]
        self.stack+=[nn.BatchNorm2d(256)]
        self.stack+=[nn.ReLU(True)]
        self.stack+=[nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False)]
        self.stack+=[nn.BatchNorm2d(128)]
        self.stack+=[nn.ReLU(True)]
        self.stack+=[nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)]
        self.stack+=[nn.BatchNorm2d(64)]
        self.stack+=[nn.ReLU(True)]
        self.stack+=[nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False)]
        self.stack+=[nn.Tanh()]
        
        self.stack=nn.Sequential(*self.stack)

    def forward(self,input):
        return self.stack(input)

class discriminator(network):
    def __init__(self):
        network.__init__(self,self)
        self.stack=[]
        self.stack+=[nn.Conv2d(3, 64, 4, 2, 1, bias=False)]
        self.stack+=[nn.LeakyReLU(0.2, inplace=True)]

        self.stack+=[nn.Conv2d(64, 128, 4, 2, 1, bias=False)]
        self.stack+=[nn.BatchNorm2d(128)]
        self.stack+=[nn.LeakyReLU(0.2, inplace=True)]
        
        self.stack+=[nn.Conv2d(128, 256, 4, 2, 1, bias=False)]
        self.stack+=[nn.BatchNorm2d(256)]
        self.stack+=[nn.LeakyReLU(0.2, inplace=True)]

        self.stack+=[nn.Conv2d(256,512, 4, 2, 1, bias=False)]
        self.stack+=[nn.BatchNorm2d(512)]
        self.stack+=[nn.LeakyReLU(0.2, inplace=True)]

        self.stack+=[nn.Conv2d(512, 1, 4, 1, 0, bias=False)]
        self.stack+=[nn.Sigmoid()]
        
        self.stack=nn.Sequential(*self.stack)
       
    def forward(self,input):
        return self.stack(input)

        



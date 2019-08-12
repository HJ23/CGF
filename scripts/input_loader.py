import torchvision.transforms as transform
import torchvision.datasets as dtst
import torch.utils as utils
from config import configs


def initialize_dataloader():
    transformation=transform.Compose([transform.ToTensor(),transform.Normalize(mean=configs.mean,std=configs.std)])
    traindata=dtst.ImageFolder(root=configs.train_data_path,transform=transformation)
    traindataloader=utils.data.DataLoader(traindata,batch_size=configs.batch_size,shuffle=configs.shuffle_data,num_workers=configs.number_of_threads)
    return traindataloader




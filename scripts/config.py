
class configs:
    learning_rate=0.0002  #learning rate
    betas=(0.5,0.999)   # beta values for adam
    batch_size=4   #batch size
    image_size=64   # input size 
    epoch=100      # epoch count
    train_data_path="../data/"   #path to image
    shuffle_data=True   # randomly shuffles images 
    number_of_threads=2   # number of threads for dataloader (option)
    mean=[0.5,0.5,0.5]   # normalize image before passing net
    std=[0.5,0.5,0.5]
    device='cpu'    # train on
    save_pretrained_epochnum=2   # save weights after this epoch count  (usefull in crash casess)  
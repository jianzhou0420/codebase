from script import *



# Functions list:
# train_cifar():
# train_cifar_from_ckpt(model_path)
# sample_cifar_from_ckpt(model_path,save,batch_size,num_batch,image_size=32)
# continue_to_train(batch_size,path)


if __name__=='__main__':
    pass
    # train_cifar(batch_size=20,epoch=100)
    # train_cifar_from_ckpt('/home/jian/git_all/codebase/DDPM/saved_models/test98.ckpt',batch_size=20,epoch=100)
    # sample_cifar_from_ckpt('/home/jian/git_all/codebase/DDPM/saved_models/test98.ckpt',save=True,batch_size=20,num_batch=1,image_size=32)
    # train_cifar()
    continue_to_train(20,200,'/home/jian/git_all/codebase/DDPM/saved_models/test188.ckpt')
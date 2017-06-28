from autoencoder import main


reg = 0.0
drop = 0.0
optim = 'adam' # 'adam'-> default lr:0.001; 'sgd'-> default lr:0.01
batchsize = 64
act = 'tanh'
max_epochs = 500
train_split = 0.9

lr = 0.001
lr_sched = [None,None] # [1,'pow2']-> lr/pow(2,epoch); [2,'pow2']->every 2nd time
lr_plat = [] # ReduceLROnPlateau: [factor, patience]

in_path = '../../data/prediction/new_go_df.pckl'
save_path = '../../data/prediction/encoded_new_go_tanh_512.npy'

struct = [4096,1024,512]



main(reg, drop, optim, batchsize, act, max_epochs, lr, train_split, lr_sched, lr_plat,
		in_path, save_path, struct)

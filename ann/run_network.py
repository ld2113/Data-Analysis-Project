from neural_network import main_emb


reg = 1e-6
drop = 0.25
optim = 'sgd' # 'adam'-> default lr:0.001; 'sgd'-> default lr:0.01
batchsize = 64
act = 'relu'
coord_norm = 'man' # 'man'-> -0.5,*2; 'mean'-> -mean, /std; None
max_epochs = 10

lr = 0.01
lr_sched = [None,None] # [1,'pow2']-> lr/pow(2,epoch); [2,'pow2']->every 2nd time
lr_plat = [] # ReduceLROnPlateau: [factor, patience]

dom_path = '../../data/encoded_dom_512_tanh.npy'
go_path = '../../data/encoded_go_tanh_512.npy'
coord_path = '../../data/080_embed/4D_coord_4x16k.npy'
train_path = '../../data/080_embed/nl_train_02_emb.npy'
test_path = '../../data/080_embed/nl_val_01_emb.npy'
#alle ramining cdg schon gemacht24.6:
input_mode = 'g' # 'c'-> coordinates only; 'd'-> domains only; 'g'-> go terms only; 'cdg'-> all three
coord_struct = [64]
dom_struct = [64]
go_struct = [64]
concat_struct = [64]
aux_output_weights = [] #[output, coord_out, dom_out, go_out]
class_weight = {0:0.08, 1:1.0}



main_emb(reg, drop, optim, batchsize, act, coord_norm, max_epochs, lr, lr_sched, lr_plat,
		dom_path, coord_path, go_path, train_path, test_path, input_mode, coord_struct, dom_struct, go_struct, concat_struct, aux_output_weights, class_weight)

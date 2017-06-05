from neural_network_emb import main_emb


reg = 0.0
drop = 0.0
optim = 'adam' # 'adam'-> default lr:0.001; 'sgd'-> default lr:0.01
batchsize = 64
act = 'relu'
coord_norm = 'man' # 'man'-> -0.5,*2; 'mean'-> -mean, /std; None
max_epochs = 5

lr = 0.001
lr_sched = None # 'pow2'-> lr/pow(2,epoch)
lr_plat = [] # ReduceLROnPlateau: [factor, patience]

dom_path = 'arrays/encoded_dom_64_tanh_1.npy'
coord_path = 'arrays/090_embed/4D_coord_4x16k.npy'
train_path = 'arrays/090_embed/nl_train_res_01_emb.npy'
test_path = 'arrays/090_embed/nl_test_01_emb.npy'

input_mode = 'cd' # 'c'-> coordinates only; 'd'-> domains only; 'cd'-> both
coord_struct = [128]
dom_struct = [128]
concat_struct = [128]
aux_output_weights = []
class_weight = {0:1.0, 1:1.0}




main_emb(reg, drop, optim, batchsize, act, coord_norm, max_epochs, lr, lr_sched, lr_plat,
		dom_path, coord_path, train_path, test_path, input_mode, coord_struct, dom_struct, concat_struct, aux_output_weights, class_weight)

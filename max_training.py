import uproot as up
import pickle

with open('/home/private/Herd/new_batcher/batched_dataset/training/train_file_list.pkl','rb')as f:
    file_list=pickle.load(f)

with open('/home/private/Herd/new_batcher/batched_dataset/training/train_ev_list.pkl','rb')as f:
    ev_list=pickle.load(f)


import warnings
warnings.filterwarnings('ignore')

massimo=0
for i,batch in enumerate(file_list):
    for j,file in enumerate(batch):
        with up.open(file) as f:
            dataframe=f['showersTree;1'].arrays('deps2D', library='pd')[ev_list[i][j][0]*400:ev_list[i][j][1]*400]
            maxs=dataframe.max(axis=0)
            if i==0 and j==0:
                massimo=maxs.max()
            if maxs.max()>massimo: massimo=maxs.max()

with open('massimo_train.pkl', 'wb') as f:
    pickle.dump(massimo, f)


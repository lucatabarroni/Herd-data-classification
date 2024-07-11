import uproot as up
import pickle
import pandas as pd
import numpy as np
from operator import itemgetter
import random

import warnings
warnings.filterwarnings('ignore')

with open('massimo_train.pkl', 'rb') as f:
    train_massimo = pickle.load(f)

with open('massimo_test.pkl', 'rb') as f:
    test_massimo = pickle.load(f)

with open('massimo_validation.pkl', 'rb') as f:
    validation_massimo = pickle.load(f)

with open('/home/private/Herd/new_batcher/batched_dataset/training/train_file_list.pkl','rb')as f:
    train_file_list=pickle.load(f)

with open('/home/private/Herd/new_batcher/batched_dataset/training/train_ev_list.pkl','rb')as f:
    train_ev_list=pickle.load(f)


with open('/home/private/Herd/new_batcher/batched_dataset/test/test_file_list.pkl','rb')as f:
    test_file_list=pickle.load(f)

with open('/home/private/Herd/new_batcher/batched_dataset/test/test_ev_list.pkl','rb')as f:
    test_ev_list=pickle.load(f)


with open('/home/private/Herd/new_batcher/batched_dataset/validation/validation_file_list.pkl','rb')as f:
    validation_file_list=pickle.load(f)

with open('/home/private/Herd/new_batcher/batched_dataset/validation/validation_ev_list.pkl','rb')as f:
    validation_ev_list=pickle.load(f)

print(len(train_file_list))
print(len(test_file_list))
print(len(validation_file_list))

for i,batch in enumerate(train_file_list):
    matr_batch=[]
    batch_labels=[]
    if i%10==0:print(i)
    for j,file in enumerate(batch):
        if file[44:49]=='proto': 
            labels=[0]*(train_ev_list[i][j][1]-train_ev_list[i][j][0])
            batch_labels=batch_labels+labels
        elif file[44:49]=='elect': 
            labels=[1]*(train_ev_list[i][j][1]-train_ev_list[i][j][0])
            batch_labels=batch_labels+labels
        else:print('errore')
        with up.open(file) as f:
            dataframe=f['showersTree;1'].arrays('deps2D', library='pd')[train_ev_list[i][j][0]*400:train_ev_list[i][j][1]*400]
            dataframe=dataframe.apply(lambda x: x/train_massimo)
            matr_file=[]
            for k in range(train_ev_list[i][j][1]-train_ev_list[i][j][0]):
                matr_file.append(np.array(dataframe.iloc[k*400:(k+1)*400]).reshape(20, 20))

            matr_batch=matr_batch+matr_file
    
    indices = np.arange(len(batch_labels))
    np.random.shuffle(indices)

    matr_batch=itemgetter(*indices)(matr_batch)
    batch_labels=itemgetter(*indices)(batch_labels)

    matr_batch = list(matr_batch)
    batch_labels = list(batch_labels)

    if len(indices)>4096:
        num_to_delete=len(indices)-4096
        indices_to_delete = random.sample(range(len(indices)), num_to_delete)
        indices_to_delete.sort()
    
        for k in range(len(indices_to_delete)):
            del matr_batch[indices_to_delete[k]]
            del batch_labels[indices_to_delete[k]]
            indices_to_delete=[x-1 for x in indices_to_delete]
                                    

    with open('/home/private/Herd/new_batcher/batched_dataset/training/bs_4096/labels/'+str(i)+'.pkl','wb') as f:
        pickle.dump(batch_labels,f)

    with open('/home/private/Herd/new_batcher/batched_dataset/training/bs_4096/data/'+str(i)+'.pkl','wb') as f:
        pickle.dump(matr_batch,f)

for i,batch in enumerate(test_file_list):
    matr_batch=[]
    batch_labels=[]
    if i%10==0:print(i)
    for j,file in enumerate(batch):
        if file[44:49]=='proto': 
            labels=[0]*(test_ev_list[i][j][1]-test_ev_list[i][j][0])
            batch_labels=batch_labels+labels
        elif file[44:49]=='elect': 
            labels=[1]*(test_ev_list[i][j][1]-test_ev_list[i][j][0])
            batch_labels=batch_labels+labels
        else:print('errore')
        with up.open(file) as f:
            dataframe=f['showersTree;1'].arrays('deps2D', library='pd')[test_ev_list[i][j][0]*400:test_ev_list[i][j][1]*400]
            dataframe=dataframe.apply(lambda x: x/test_massimo)
            matr_file=[]
            for k in range(test_ev_list[i][j][1]-test_ev_list[i][j][0]):
                matr_file.append(np.array(dataframe.iloc[k*400:(k+1)*400]).reshape(20, 20))

            matr_batch=matr_batch+matr_file
    
    indices = np.arange(len(batch_labels))
    np.random.shuffle(indices)

    matr_batch=itemgetter(*indices)(matr_batch)
    batch_labels=itemgetter(*indices)(batch_labels)

    matr_batch = list(matr_batch)
    batch_labels = list(batch_labels)

    if len(indices)>4096:
        num_to_delete=len(indices)-4096
        indices_to_delete = random.sample(range(len(indices)), num_to_delete)
        indices_to_delete.sort()
    
        for k in range(len(indices_to_delete)):
            del matr_batch[indices_to_delete[k]]
            del batch_labels[indices_to_delete[k]]
            indices_to_delete=[x-1 for x in indices_to_delete]

    with open('/home/private/Herd/new_batcher/batched_dataset/test/bs_4096/labels/'+str(i)+'.pkl','wb') as f:
        pickle.dump(batch_labels,f)

    with open('/home/private/Herd/new_batcher/batched_dataset/test/bs_4096/data/'+str(i)+'.pkl','wb') as f:
        pickle.dump(matr_batch,f)

for i,batch in enumerate(validation_file_list):
    matr_batch=[]
    batch_labels=[]
    if i%10==0:print(i)
    for j,file in enumerate(batch):
        if file[44:49]=='proto': 
            labels=[0]*(validation_ev_list[i][j][1]-validation_ev_list[i][j][0])
            batch_labels=batch_labels+labels
        elif file[44:49]=='elect': 
            labels=[1]*(validation_ev_list[i][j][1]-validation_ev_list[i][j][0])
            batch_labels=batch_labels+labels
        else:print('errore')
        with up.open(file) as f:
            dataframe=f['showersTree;1'].arrays('deps2D', library='pd')[validation_ev_list[i][j][0]*400:validation_ev_list[i][j][1]*400]
            dataframe=dataframe.apply(lambda x: x/validation_massimo)
            matr_file=[]
            for k in range(validation_ev_list[i][j][1]-validation_ev_list[i][j][0]):
                matr_file.append(np.array(dataframe.iloc[k*400:(k+1)*400]).reshape(20, 20))

            matr_batch=matr_batch+matr_file
    
    indices = np.arange(len(batch_labels))
    np.random.shuffle(indices)

    matr_batch=itemgetter(*indices)(matr_batch)
    batch_labels=itemgetter(*indices)(batch_labels)

    matr_batch = list(matr_batch)
    batch_labels = list(batch_labels)

    if len(indices)>4096:
        num_to_delete=len(indices)-4096
        indices_to_delete = random.sample(range(len(indices)), num_to_delete)
        indices_to_delete.sort()
    
        for k in range(len(indices_to_delete)):
            del matr_batch[indices_to_delete[k]]
            del batch_labels[indices_to_delete[k]]
            indices_to_delete=[x-1 for x in indices_to_delete]

    with open('/home/private/Herd/new_batcher/batched_dataset/validation/bs_4096/labels/'+str(i)+'.pkl','wb') as f:
        pickle.dump(batch_labels,f)

    with open('/home/private/Herd/new_batcher/batched_dataset/validation/bs_4096/data/'+str(i)+'.pkl','wb') as f:
        pickle.dump(matr_batch,f)
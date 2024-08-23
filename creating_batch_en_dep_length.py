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

#################################
#CAMBIA QUESTE DUE RIGHE PER CAMBIARE LA DIMENSIONE DEL BATCHING
bs_dataset='bs_4096'
bs=4096
#################################

with open('/home/private/Herd/new_batcher/batched_dataset/training/'+bs_dataset+'/train_file_list.pkl','rb')as f:
    train_file_list=pickle.load(f)

with open('/home/private/Herd/new_batcher/batched_dataset/training/'+bs_dataset+'/train_ev_list.pkl','rb')as f:
    train_ev_list=pickle.load(f)


with open('/home/private/Herd/new_batcher/batched_dataset/test/'+bs_dataset+'/test_file_list.pkl','rb')as f:
    test_file_list=pickle.load(f)

with open('/home/private/Herd/new_batcher/batched_dataset/test/'+bs_dataset+'/test_ev_list.pkl','rb')as f:
    test_ev_list=pickle.load(f)


with open('/home/private/Herd/new_batcher/batched_dataset/validation/'+bs_dataset+'/validation_file_list.pkl','rb')as f:
    validation_file_list=pickle.load(f)

with open('/home/private/Herd/new_batcher/batched_dataset/validation/'+bs_dataset+'/validation_ev_list.pkl','rb')as f:
    validation_ev_list=pickle.load(f)

print(len(train_file_list))
print(len(test_file_list))
print(len(validation_file_list))

for i,batch in enumerate(train_file_list):
    matr_batch=[]
    batch_labels=[]
    en_dep=[]
    length=[]
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
            en_dep_dataframe=f['showersTree;1'].arrays('totDep', library='pd')[train_ev_list[i][j][0]:train_ev_list[i][j][1]]
            en_dep=en_dep+en_dep_dataframe['totDep'].tolist()
            length_dataframe=f['showersTree;1'].arrays('trackLengthLYSOX0', library='pd')[train_ev_list[i][j][0]:train_ev_list[i][j][1]]
            length=length+length_dataframe['trackLengthLYSOX0'].tolist()
            
            matr_file=[]
            for k in range(train_ev_list[i][j][1]-train_ev_list[i][j][0]):
                matr_file.append(np.array(dataframe.iloc[k*400:(k+1)*400]).reshape(20, 20))

            matr_batch=matr_batch+matr_file

    indices = np.arange(len(batch_labels))
    np.random.shuffle(indices)

    matr_batch=itemgetter(*indices)(matr_batch)
    batch_labels=itemgetter(*indices)(batch_labels)
    en_dep=itemgetter(*indices)(en_dep)
    length=itemgetter(*indices)(length)

    matr_batch = list(matr_batch)
    batch_labels = list(batch_labels)
    en_dep=list(en_dep)
    length=list(length)

    if len(indices)>bs:
        num_to_delete=len(indices)-bs
        indices_to_delete = random.sample(range(len(indices)), num_to_delete)
        indices_to_delete.sort()
    
        for k in range(len(indices_to_delete)):
            del matr_batch[indices_to_delete[k]]
            del batch_labels[indices_to_delete[k]]
            del en_dep[indices_to_delete[k]]
            del length[indices_to_delete[k]]
            indices_to_delete=[x-1 for x in indices_to_delete]
                                    

    with open('/home/private/Herd/new_batcher/test_en_dep/training/'+bs_dataset+'/labels/'+str(i)+'.pkl','wb') as f:
        pickle.dump(batch_labels,f)

    with open('/home/private/Herd/new_batcher/test_en_dep/training/'+bs_dataset+'/data/'+str(i)+'.pkl','wb') as f:
        pickle.dump(matr_batch,f)

    with open('/home/private/Herd/new_batcher/test_en_dep/training/'+bs_dataset+'/dep_energy/'+str(i)+'.pkl','wb') as f:
        pickle.dump(en_dep,f)

    with open('/home/private/Herd/new_batcher/test_en_dep/training/'+bs_dataset+'/length/'+str(i)+'.pkl','wb') as f:
        pickle.dump(length,f)

for i,batch in enumerate(test_file_list):
    matr_batch=[]
    batch_labels=[]
    en_dep=[]
    length=[]
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
            en_dep_dataframe=f['showersTree;1'].arrays('totDep', library='pd')[test_ev_list[i][j][0]:test_ev_list[i][j][1]]
            en_dep=en_dep+en_dep_dataframe['totDep'].tolist()
            length_dataframe=f['showersTree;1'].arrays('trackLengthLYSOX0', library='pd')[test_ev_list[i][j][0]:test_ev_list[i][j][1]]
            length=length+length_dataframe['trackLengthLYSOX0'].tolist()
            
            matr_file=[]
            for k in range(test_ev_list[i][j][1]-test_ev_list[i][j][0]):
                matr_file.append(np.array(dataframe.iloc[k*400:(k+1)*400]).reshape(20, 20))

            matr_batch=matr_batch+matr_file
    
    indices = np.arange(len(batch_labels))
    np.random.shuffle(indices)

    matr_batch=itemgetter(*indices)(matr_batch)
    batch_labels=itemgetter(*indices)(batch_labels)
    en_dep=itemgetter(*indices)(en_dep)
    length=itemgetter(*indices)(length)

    matr_batch = list(matr_batch)
    batch_labels = list(batch_labels)
    en_dep=list(en_dep)
    length=list(length)

    if len(indices)>bs:
        num_to_delete=len(indices)-bs
        indices_to_delete = random.sample(range(len(indices)), num_to_delete)
        indices_to_delete.sort()
    
        for k in range(len(indices_to_delete)):
            del matr_batch[indices_to_delete[k]]
            del batch_labels[indices_to_delete[k]]
            del en_dep[indices_to_delete[k]]
            del length[indices_to_delete[k]]
            indices_to_delete=[x-1 for x in indices_to_delete]

    with open('/home/private/Herd/new_batcher/test_en_dep/test/'+bs_dataset+'/labels/'+str(i)+'.pkl','wb') as f:
        pickle.dump(batch_labels,f)

    with open('/home/private/Herd/new_batcher/test_en_dep/test/'+bs_dataset+'/data/'+str(i)+'.pkl','wb') as f:
        pickle.dump(matr_batch,f)

    with open('/home/private/Herd/new_batcher/test_en_dep/test/'+bs_dataset+'/dep_energy/'+str(i)+'.pkl','wb') as f:
        pickle.dump(en_dep,f)

    with open('/home/private/Herd/new_batcher/test_en_dep/test/'+bs_dataset+'/length/'+str(i)+'.pkl','wb') as f:
        pickle.dump(length,f)
        
for i,batch in enumerate(validation_file_list):
    matr_batch=[]
    batch_labels=[]
    en_dep=[]
    length=[]
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
            en_dep_dataframe=f['showersTree;1'].arrays('totDep', library='pd')[validation_ev_list[i][j][0]:validation_ev_list[i][j][1]]
            en_dep=en_dep+en_dep_dataframe['totDep'].tolist()
            length_dataframe=f['showersTree;1'].arrays('trackLengthLYSOX0', library='pd')[validation_ev_list[i][j][0]:validation_ev_list[i][j][1]]
            length=length+length_dataframe['trackLengthLYSOX0'].tolist()
            
            matr_file=[]
            for k in range(validation_ev_list[i][j][1]-validation_ev_list[i][j][0]):
                matr_file.append(np.array(dataframe.iloc[k*400:(k+1)*400]).reshape(20, 20))

            matr_batch=matr_batch+matr_file
    
    indices = np.arange(len(batch_labels))
    np.random.shuffle(indices)

    matr_batch=itemgetter(*indices)(matr_batch)
    batch_labels=itemgetter(*indices)(batch_labels)
    en_dep=itemgetter(*indices)(en_dep)
    length=itemgetter(*indices)(length)

    matr_batch = list(matr_batch)
    batch_labels = list(batch_labels)
    en_dep=list(en_dep)
    length=list(length)

    if len(indices)>bs:
        num_to_delete=len(indices)-bs
        indices_to_delete = random.sample(range(len(indices)), num_to_delete)
        indices_to_delete.sort()
    
        for k in range(len(indices_to_delete)):
            del matr_batch[indices_to_delete[k]]
            del batch_labels[indices_to_delete[k]]
            del en_dep[indices_to_delete[k]]
            del length[indices_to_delete[k]]
            indices_to_delete=[x-1 for x in indices_to_delete]

    with open('/home/private/Herd/new_batcher/test_en_dep/validation/'+bs_dataset+'/labels/'+str(i)+'.pkl','wb') as f:
        pickle.dump(batch_labels,f)

    with open('/home/private/Herd/new_batcher/test_en_dep/validation/'+bs_dataset+'/data/'+str(i)+'.pkl','wb') as f:
        pickle.dump(matr_batch,f)

    with open('/home/private/Herd/new_batcher/test_en_dep/validation/'+bs_dataset+'/dep_energy/'+str(i)+'.pkl','wb') as f:
        pickle.dump(en_dep,f)

    with open('/home/private/Herd/new_batcher/test_en_dep/validation/'+bs_dataset+'/length/'+str(i)+'.pkl','wb') as f:
        pickle.dump(length,f)





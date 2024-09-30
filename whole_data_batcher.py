import uproot as up
import numpy as np
import pickle
import warnings
import os
import pandas as pd
import random
warnings.filterwarnings('ignore')

with open('/home/private/Herd/datasets/dataset1/informations/train_file_list.pkl','rb')as f:
    train_file_list=pickle.load(f)

with open('/home/private/Herd/datasets/dataset1/informations/train_ev_list.pkl','rb')as f:
    train_ev_list=pickle.load(f)

with open('/home/private/Herd/datasets/dataset1/informations/test_file_list.pkl','rb')as f:
    test_file_list=pickle.load(f)

with open('/home/private/Herd/datasets/dataset1/informations/test_ev_list.pkl','rb')as f:
    test_ev_list=pickle.load(f)

with open('/home/private/Herd/datasets/dataset1/informations/validation_file_list.pkl','rb')as f:
    validation_file_list=pickle.load(f)

with open('/home/private/Herd/datasets/dataset1/informations/validation_ev_list.pkl','rb')as f:
    validation_ev_list=pickle.load(f)

num_train_ev_to_move=len(train_ev_list[0])-4096

train_ev_to_move=[]
for i in range(num_train_ev_to_move):
    train_ev_to_move.append(random.randint(0,len(train_ev_list[0])))

for i in range(len(train_ev_list)-1):
    mov_ev=train_ev_to_move
    for ev in mov_ev:
        train_ev_list[-1].append(train_ev_list[i][ev])
        train_file_list[-1].append(train_file_list[i][ev])
        train_ev_list[i].pop(ev)
        train_file_list[i].pop(ev)
        mov_ev=[el-1 for el in mov_ev]

num_test_ev_to_move=len(test_ev_list[0])-4096

test_ev_to_move=[]
for i in range(num_test_ev_to_move):
    test_ev_to_move.append(random.randint(0,len(test_ev_list[0])))

for i in range(len(test_ev_list)-1):
    mov_ev=test_ev_to_move
    for ev in mov_ev:
        test_ev_list[-1].append(test_ev_list[i][ev])
        test_file_list[-1].append(test_file_list[i][ev])
        test_ev_list[i].pop(ev)
        test_file_list[i].pop(ev)
        mov_ev=[el-1 for el in mov_ev]

num_validation_ev_to_move=len(validation_ev_list[0])-4096

validation_ev_to_move=[]
for i in range(num_validation_ev_to_move):
    validation_ev_to_move.append(random.randint(0,len(validation_ev_list[0])))

for i in range(len(validation_ev_list)-1):
    mov_ev=validation_ev_to_move
    for ev in mov_ev:
        validation_ev_list[-1].append(validation_ev_list[i][ev])
        validation_file_list[-1].append(validation_file_list[i][ev])
        validation_ev_list[i].pop(ev)
        validation_file_list[i].pop(ev)
        mov_ev=[el-1 for el in mov_ev]

files_list=train_file_list+test_file_list+validation_file_list
ev_list=train_ev_list+test_ev_list+validation_ev_list
sets=['tr']*len(train_file_list)+['te']*len(test_file_list)+['va']*len(validation_file_list)

positions={}
batch_to_sub=0
for i, batch in enumerate(files_list):
    if i>=len(train_file_list)+len(test_file_list):
        batch_to_sub=len(train_file_list)+len(test_file_list)
    elif i>=len(train_file_list):
        batch_to_sub=len(train_file_list)
    for j, file in enumerate(batch):
        if file[44:46]=='el': label=1
        elif file[44:46]=='pr': label=0
        else:
            print(file[44:46])
            raise Exception("Neither singal nor background")
        if file in positions:
            positions[file].append(((ev_list[i][j]),sets[i],i-batch_to_sub,label))
        else:
            positions[file]=[((ev_list[i][j]),sets[i],i-batch_to_sub,label)]
with open('file_events_positions.pkl','wb') as f:
    pickle.dump(positions,f)

files=list(positions.keys())

train_batched_data=[]
train_batched_labels=[]
train_batch_id=[]
train_batched_en_dep=[]
train_batched_len=[]
train_batched_e0=[]
for i in range(len(train_ev_list)):
    train_batched_data.append([])
    train_batched_labels.append([])
    train_batched_en_dep.append([])
    train_batched_len.append([])
    train_batched_e0.append([])
    train_batch_id.append([])

test_batched_data=[]
test_batched_labels=[]
test_batch_id=[]
test_batched_en_dep=[]
test_batched_len=[]
test_batched_e0=[]
for i in range(len(test_ev_list)):
    test_batched_data.append([])
    test_batched_labels.append([])
    test_batched_en_dep.append([])
    test_batched_len.append([])
    test_batched_e0.append([])
    test_batch_id.append([])

validation_batched_data=[]
validation_batched_labels=[]
validation_batch_id=[]
validation_batched_en_dep=[]
validation_batched_len=[]
validation_batched_e0=[]
for i in range(len(validation_ev_list)):
    validation_batched_data.append([])
    validation_batched_labels.append([])
    validation_batched_en_dep.append([])
    validation_batched_len.append([])
    validation_batched_e0.append([])
    validation_batch_id.append([])

for i in range(len(train_ev_list)):
    with open('train/train_data/'+str(i)+'.pkl','wb') as f:
        pickle.dump([],f)
    with open('train/train_labels/'+str(i)+'.pkl','wb') as f:
        pickle.dump([],f)
    with open('train/train_en_dep/'+str(i)+'.pkl','wb') as f:
        pickle.dump([],f)
    with open('train/train_len/'+str(i)+'.pkl','wb') as f:
        pickle.dump([],f)
    with open('train/train_e0/'+str(i)+'.pkl','wb') as f:
        pickle.dump([],f)
    

for i in range(len(test_ev_list)):
    with open('test/test_data/'+str(i)+'.pkl','wb') as f:
        pickle.dump([],f)
    with open('test/test_labels/'+str(i)+'.pkl','wb') as f:
        pickle.dump([],f)
    with open('test/test_en_dep/'+str(i)+'.pkl','wb') as f:
        pickle.dump([],f)
    with open('test/test_len/'+str(i)+'.pkl','wb') as f:
        pickle.dump([],f)
    with open('test/test_e0/'+str(i)+'.pkl','wb') as f:
        pickle.dump([],f)

for i in range(len(validation_ev_list)):
    with open('validation/validation_data/'+str(i)+'.pkl','wb') as f:
        pickle.dump([],f)
    with open('validation/validation_labels/'+str(i)+'.pkl','wb') as f:
        pickle.dump([],f)
    with open('validation/validation_en_dep/'+str(i)+'.pkl','wb') as f:
        pickle.dump([],f)
    with open('validation/validation_len/'+str(i)+'.pkl','wb') as f:
        pickle.dump([],f)
    with open('validation/validation_e0/'+str(i)+'.pkl','wb') as f:
        pickle.dump([],f)


massimo_train=-1
massimo_test=-1
massimo_validation=-1

print(len(files))

for k,file in enumerate(files):
    with up.open(file) as f:
        events_id=[]
        sets_id=[]
        labels=[]
        batches_id=[]
        dataframe=f['showersTree;1'].arrays('deps2D', library='pd')
        en_dataframe=f['showersTree;1'].arrays('totDep', library='pd')
        len_dataframe=f['showersTree;1'].arrays('trackLengthLYSOX0', library='pd')
        eo_dataframe=f['showersTree;1'].arrays('E0', library='pd')
        for tupla in positions[file]:    
            events_id.append(tupla[0])
            sets_id.append(tupla[1])
            batches_id.append(tupla[2])
            labels.append(tupla[3])
            
        for i,ev_id in enumerate(events_id):
            event=dataframe.loc[ev_id]['deps2D']
            tot_dep=en_dataframe.loc[ev_id]['totDep']
            length=len_dataframe.loc[ev_id]['trackLengthLYSOX0']
            eo=eo_dataframe.loc[ev_id]['E0']
            if sets_id[i]=='tr':
                if (massimo_train<event.max()):
                    massimo_train=event.max()
                    
                train_batched_data[batches_id[i]].append(np.array(event).reshape(20,20))
                train_batched_labels[batches_id[i]].append(labels[i])
                train_batched_en_dep[batches_id[i]].append(np.array(tot_dep))
                train_batched_len[batches_id[i]].append(np.array(length))
                train_batched_e0[batches_id[i]].append(np.array(eo))
                train_batch_id[batches_id[i]].append(batches_id[i])
        
            elif sets_id[i]=='te':
                if (massimo_test<event.max()):
                    massimo_test=event.max()
                    
                test_batched_data[batches_id[i]].append(np.array(event).reshape(20,20))
                test_batched_labels[batches_id[i]].append(labels[i])
                test_batched_en_dep[batches_id[i]].append(np.array(tot_dep))
                test_batched_len[batches_id[i]].append(np.array(length))
                test_batched_e0[batches_id[i]].append(np.array(eo))
                test_batch_id[batches_id[i]].append(batches_id[i])
                    
            elif sets_id[i]=='va':
                if (massimo_validation<event.max()):
                    massimo_validation=event.max()

                validation_batched_data[batches_id[i]].append(np.array(event).reshape(20,20))
                validation_batched_labels[batches_id[i]].append(labels[i])
                validation_batched_en_dep[batches_id[i]].append(np.array(tot_dep))
                validation_batched_len[batches_id[i]].append(np.array(length))
                validation_batched_e0[batches_id[i]].append(np.array(eo))
                validation_batch_id[batches_id[i]].append(batches_id[i])
    
            else: raise Exception("Not an acceptable set")

    if k%1000==0 or k==len(files)-1:
        print('la len di train_bathced_data è: '+str(len(train_batched_data)))
        print('Sto salvando...')
        for i,batch in enumerate(train_batched_data):
            if len(batch)!=0:
                if i%50==0:
                    print('il batch '+str(i)+' del train')
                with open('train/train_data/'+str(i)+'.pkl','rb') as f:
                    train_data=pickle.load(f)
                train_data.extend(batch)
                with open('train/train_data/'+str(i)+'.pkl','wb') as f:
                    pickle.dump(train_data,f)
                    
                with open('train/train_labels/'+str(i)+'.pkl','rb') as f:
                    train_labels=pickle.load(f)
                train_labels.extend(train_batched_labels[i])
                with open('train/train_labels/'+str(i)+'.pkl','wb') as f:
                    pickle.dump(train_labels,f)
                
                with open('train/train_en_dep/'+str(i)+'.pkl','rb') as f:
                    train_en_dep=pickle.load(f)
                train_en_dep.extend(train_batched_en_dep[i])
                with open('train/train_en_dep/'+str(i)+'.pkl','wb') as f:
                    pickle.dump(train_en_dep,f)
                
                with open('train/train_len/'+str(i)+'.pkl','rb') as f:
                    train_len=pickle.load(f)
                train_len.extend(train_batched_len[i])
                with open('train/train_len/'+str(i)+'.pkl','wb') as f:
                    pickle.dump(train_len,f)
                
                with open('train/train_e0/'+str(i)+'.pkl','rb') as f:
                    train_e0=pickle.load(f)
                train_e0.extend(train_batched_e0[i])
                with open('train/train_e0/'+str(i)+'.pkl','wb') as f:
                    pickle.dump(train_e0,f)
                    
                train_data=[]
                train_labels=[]    
                train_batched_data[i]=[]
                train_batched_labels[i]=[]
                train_batch_id[i]=[]
                train_batched_en_dep[i]=[]
                train_batched_len[i]=[]
                train_batched_e0[i]=[]
        print('la len di test_bathced_data è: '+str(len(test_batched_data)))
        print('Sto salvando...')
        for i,batch in enumerate(test_batched_data):
            if len(batch)!=0:
                if i%50==0:
                    print('il batch '+str(i)+' del test')
                with open('test/test_data/'+str(i)+'.pkl','rb') as f:
                    test_data=pickle.load(f)
                test_data.extend(batch)
                with open('test/test_data/'+str(i)+'.pkl','wb') as f:
                    pickle.dump(test_data,f)
    
                with open('test/test_labels/'+str(i)+'.pkl','rb') as f:
                    test_labels=pickle.load(f)
                test_labels.extend(test_batched_labels[i])
                with open('test/test_labels/'+str(i)+'.pkl','wb') as f:
                    pickle.dump(test_labels,f)

                with open('test/test_en_dep/'+str(i)+'.pkl','rb') as f:
                    test_en_dep=pickle.load(f)
                test_en_dep.extend(test_batched_en_dep[i])
                with open('test/test_en_dep/'+str(i)+'.pkl','wb') as f:
                    pickle.dump(test_en_dep,f)

                with open('test/test_len/'+str(i)+'.pkl','rb') as f:
                    test_len=pickle.load(f)
                test_len.extend(test_batched_len[i])
                with open('test/test_len/'+str(i)+'.pkl','wb') as f:
                    pickle.dump(test_len,f)

                with open('test/test_e0/'+str(i)+'.pkl','rb') as f:
                    test_e0=pickle.load(f)
                test_e0.extend(test_batched_e0[i])
                with open('test/test_e0/'+str(i)+'.pkl','wb') as f:
                    pickle.dump(test_e0,f)
    
                test_data=[]
                test_labels=[]    
                test_batched_data[i]=[]
                test_batched_labels[i]=[]
                test_batch_id[i]=[]
                test_batched_en_dep[i]=[]
                test_batched_len[i]=[]
                test_batched_e0[i]=[]
        print('la len di validation_bathced_data è: '+str(len(validation_batched_data)))
        print('Sto salvando...')
        for i,batch in enumerate(validation_batched_data):
            if len(batch)!=0:
                if i%50==0:
                    print('il batch '+str(i)+' del validation')
                with open('validation/validation_data/'+str(i)+'.pkl','rb') as f:
                    validation_data=pickle.load(f)
                validation_data.extend(batch)
                with open('validation/validation_data/'+str(i)+'.pkl','wb') as f:
                    pickle.dump(validation_data,f)
    
                with open('validation/validation_labels/'+str(i)+'.pkl','rb') as f:
                    validation_labels=pickle.load(f)
                validation_labels.extend(validation_batched_labels[i])
                with open('validation/validation_labels/'+str(i)+'.pkl','wb') as f:
                    pickle.dump(validation_labels,f)

                with open('validation/validation_en_dep/'+str(i)+'.pkl','rb') as f:
                    validation_en_dep=pickle.load(f)
                validation_en_dep.extend(validation_batched_en_dep[i])
                with open('validation/validation_en_dep/'+str(i)+'.pkl','wb') as f:
                    pickle.dump(validation_en_dep,f)
                
                with open('validation/validation_len/'+str(i)+'.pkl','rb') as f:
                    validation_len=pickle.load(f)
                validation_len.extend(validation_batched_len[i])
                with open('validation/validation_len/'+str(i)+'.pkl','wb') as f:
                    pickle.dump(validation_len,f)

                with open('validation/validation_e0/'+str(i)+'.pkl','rb') as f:
                    validation_e0=pickle.load(f)
                validation_e0.extend(validation_batched_e0[i])
                with open('validation/validation_e0/'+str(i)+'.pkl','wb') as f:
                    pickle.dump(validation_e0,f)
    
                validation_data=[]
                validation_labels=[]
                validation_batched_data[i]=[]
                validation_batched_labels[i]=[]
                validation_batch_id[i]=[]
                validation_batched_en_dep[i]=[]
                validation_batched_len[i]=[]
                validation_batched_e0[i]=[]
                    
    if k%10==0:
        print(k)
with open('massimo_train.pkl','wb') as f:
    pickle.dump(massimo_train,f)

with open('massimo_test.pkl','wb') as f:
    pickle.dump(massimo_test,f)

with open('massimo_validation.pkl','wb') as f:
    pickle.dump(massimo_validation,f)

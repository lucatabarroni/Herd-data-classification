import data_Loader
import pickle

dir=['/home/rgw/scratch/formato/showerpics/electrons_100GeV_1TeV/','/home/rgw/scratch/formato/showerpics/electrons_1TeV_20TeV/',
     '/home/rgw/scratch/formato/showerpics/protons_100GeV_1TeV/','/home/rgw/scratch/formato/showerpics/protons_1TeV_10TeV/']

tree_name=['showersTree;1','showersTree;1','showersTree;1','showersTree;1']


object=data_Loader.Loader(dir,tree_name,batch_size=4096,training_size=0.5,test_size=0.3,validation_size=0.2)

with open('train_file_list.pkl','wb') as f:
    pickle.dump(object.train_file_list,f)

with open('train_ev_list.pkl','wb') as f:
    pickle.dump(object.train_ev_list,f)

with open('test_file_list.pkl','wb') as f:
    pickle.dump(object.test_file_list,f)

with open('test_ev_list.pkl' , 'wb') as f:
    pickle.dump(object.test_ev_list,f)

with open('validation_file_list.pkl','wb') as f:
    pickle.dump(object.validation_file_list,f)
        
with open('validation_ev_list.pkl','wb') as f:
    pickle.dump(object.validation_ev_list,f)
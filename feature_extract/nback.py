import os
import scipy.io
import numpy


from feature_compute import dataset_feature


path="./data/nback/"
write_path="./feature/nback/"


    
def getdata(file_name):

    #读取
    full_path=path+file_name
    read_dict=scipy.io.loadmat(full_path)

    raw_data=read_dict['data']
    raw_data=raw_data[:,:7680]                                  #19201-->19200
    
    return raw_data
    
def slices(data):                                       #interval=2.3,5-2,5-3


    #切片
    newdata=[]
    for i in range(5):
        newdata.append(data[:,i*1280:1280*i+2560])
    
    print(1280*i+2560)
    newdata=numpy.stack(newdata)
    
    return newdata
    




    
if __name__=='__main__':

    
    
    
    
    list_dirs=os.listdir(path)
    for sub_path in list_dirs:
        print(sub_path)
        write_dict={}
        
        data=getdata(sub_path)

        data=dataset_feature(slices(data))
        
        write_dict['data']=data

        
        full_write_path=write_path+sub_path
        scipy.io.savemat(full_write_path,write_dict)
        
        
    
    

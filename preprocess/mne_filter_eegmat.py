import numpy
import mne
import os
import scipy.io
 
write_path="../filter/eegmat/"
read_path="./after/eegmat/"


def filters(data):
    out=[]
    for i in range(9):
        il=i*4+4
        ih=i*4+8
        chd=numpy.ones(data.shape)
        for idx in range(len(data)):
            chd[idx]=mne.filter.filter_data(data[idx],sfreq=128,l_freq=il,h_freq=ih,method='fir',fir_window='hamming',phase='zero-double',verbose='WARNING')
        out.append(chd)
    out=numpy.stack(out)
    return out

def mnePreprocess(raw_path):

    #读取
    
    read_dict=scipy.io.loadmat(read_path+raw_path)
    data=read_dict['data']
    data=data[:,:7680]

    print(raw_path)
    #载入数据
    data=filters(data)

    #保存数据
    save_dict={}    
    save_dict['data']=data
    full_write_path=write_path+sub_path
    scipy.io.savemat(full_write_path,save_dict)

    return

if __name__=="__main__":

    
    list_dirs=os.listdir(read_path)
    for sub_path in list_dirs:
        mnePreprocess(sub_path)





        


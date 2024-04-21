import numpy
#import pathlib
import mne
#from mne.preprocessing import ICA
import os
#import ast

#import pywt
import scipy.io
import matplotlib.pyplot as plt

   
write_path="./after/eegmat/"
read_path="./raw/eegmat/"
'''
def get_eog_exclude(scores):

    mean_exclude=numpy.argsort(numpy.mean(numpy.abs(scores),axis=0))[-2:]   
    #print(mean_exclude)

    max_exclude=numpy.argsort(numpy.abs(scores))[:,-1]
    #print(max_exclude)

    exclude=list(set(mean_exclude)&set(max_exclude))
    #print(exclude)

    return exclude

def get_mascle_exclude(scores):
    exclude=[]

    for idx,i in enumerate(scores):
        if i>0.9:
            exclude.append(idx)
    #print(exclude)
    return exclude
    
'''
def diff(x,n_step=1):
    d=numpy.zeros(len(x)-1,dtype=numpy.float32)
    for idx in range(len(x)):
        if idx+n_step<len(x):
            d[idx]=x[idx+n_step]-x[idx]
    return d

def denose2(data,width=1.5):
    new_data=numpy.zeros(data.shape,dtype=numpy.float32)
    
    for idx,chs_data in enumerate(data):
        d_data=diff(chs_data)
        
        score=0
        Q=numpy.percentile(abs(d_data),[0,25,50,75,100])
        IQR=Q[3]-Q[1]
        low_b=Q[1]-width*IQR                         #boundary
        high_b=Q[3]+width*IQR

        new_d=[]
        last=Q[2]                         #实际值d。修改值dd。符号，0：非离群值，1：上离群值，-1：下离群值。
        
        for d in d_data:
            if (abs(d)>high_b):
                score+=1
                new_d.append(numpy.sign(d)*last)                    
            else:
                new_d.append(d)
                last=abs(d)
                
        score=1-(score/len(d_data))
        if score<0.9:
            print("denose2",idx+1,score)
        
        new_chs=numpy.zeros(chs_data.shape,dtype=numpy.float32)
        new_chs[0]=chs_data[0]
        
        for jdx,d in enumerate(new_d):
            
            new_chs[jdx+1]=new_chs[jdx]+d
            
        new_data[idx]=numpy.stack(new_chs)
        
    return new_data

    
def denose(data,width=1.5):
    new_data=numpy.zeros(data.shape,dtype=numpy.float32)
    
    for idx,chs_data in enumerate(data):
        score=0
        Q=numpy.percentile(chs_data,[0,25,50,75,100])
        IQR=Q[3]-Q[1]
        low_b=Q[1]-width*IQR                         #boundary
        high_b=Q[3]+width*IQR
        
        new_chs=[]
        last_1=[Q[2],Q[2],0]                         #实际值d。修改值dd。符号，0：非离群值，1：上离群值，-1：下离群值。
        last_2=[(Q[1]+Q[3])/2,(Q[1]+Q[3])/2,0]
        dd=Q[2]
        for d in chs_data:
            if d>high_b:
                score+=1
                
                if last_1[-1]==1:
                    dd=d-last_1[0]+last_1[1]
                    if dd>high_b:
                        dd=max(last_1[1],last_2[1])
                    elif dd<low_b:
                        dd=min(last_1[1],last_2[1])
    
                else:
                    dd=max(last_1[1],last_2[1])
                new_chs.append(dd)
                last_2=last_1
                last_1=[d,dd,1]
                
            elif d<low_b:
                score+=1
                
                if last_1[-1]==-1:
                    dd=d-last_1[0]+last_1[1]
                    if dd>high_b:
                        dd=max(last_1[1],last_2[1])
                    elif dd<low_b:
                        dd=min(last_1[1],last_2[1])

                else:
                    dd=min(last_1[1],last_2[1])
                new_chs.append(dd)
                last_2=last_1
                last_1=[d,dd,-1]
        
            else:
                dd=d
                new_chs.append(dd)
                last_2=last_1
                last_1=[d,dd,0]
        score=1-(score/len(chs_data))
        if score<0.9:
            print("denose",idx+1,score)
        new_data[idx]=numpy.stack(new_chs)
        
    return new_data
def scale(data):
    new_data=numpy.zeros(data.shape,dtype=numpy.float32)
    for idx,chs_data in enumerate(data):
        mi=min(chs_data)
        ma=max(chs_data)
        new_data[idx]=(chs_data-mi)/(ma-mi)
    return new_data
def regular(data):
    data=denose(data)
    data=scale(data)
    
    return data
def watch(data):
    plt.show()
    for idx,i in enumerate(data):
        plt.title(idx)
        plt.plot(i)
        plt.show()
    plt.plot(numpy.stack(list(range(len(data.flatten()))))/len(data.flatten())*10,data.flatten())
    plt.show()
def reform(data):
    
    ch_names=(['F3','F7','T7','P7','O1','O2','P8','T8','F8','F4'])
    sfreq=500
    ch_types='eeg'
    info=mne.create_info(ch_names,sfreq,ch_types=ch_types,verbose='WARNING')
    raw=mne.io.RawArray(data,info,verbose='WARNING')
    #raw.set_montage('standard_1020',on_missing='raise',verbose='WARNING')
    return raw

def mnePreprocess(raw_path,stage):

    #读取
    full_read_path=read_path+raw_path
    raw=mne.io.read_raw_edf(full_read_path,verbose='WARNING')

    print(raw_path)
    
    #载入数据
    raw.load_data(verbose='WARNING')
    raw.rename_channels({'EEG Fp1':'Fp1',
                         'EEG Fp2':'Fp2',
                         'EEG F3':'F3',
                         'EEG F4':'F4',
                         'EEG F7':'F7',
                         'EEG F8':'F8',
                         'EEG T3':'T7',
                         'EEG T4':'T8',
                         'EEG C3':'C3',
                         'EEG C4':'C4',
                         'EEG T5':'P7',
                         'EEG T6':'P8',
                         'EEG P3':'P3',
                         'EEG P4':'P4',
                         'EEG O1':'O1',
                         'EEG O2':'O2',
                         'EEG Fz':'Fz',
                         'EEG Cz':'Cz',
                         'EEG Pz':'Pz',
                         'EEG A2-A1':'A2-A1',
                         'ECG ECG':'ECG'})
    
    #通道选择
    #raw.drop_channels(['Fp1','Fp2','C3','C4','P3','P4','Fz','Cz','Pz','A2-A1','ECG'])
    raw.reorder_channels(['F3','F7','T7','P7','O1','O2','P8','T8','F8','F4'])
    #print(raw.info)
    
    #channels locaion
    #montage=mne.channels.make_standard_montage('standard_1020')
    #raw.set_montage('standard_1020',on_missing='raise',verbose='WARNING')
    #raw.plot_psd_topomap()
    #raw.plot_sensors()
    #plt.show()
    
    #重参考
    #raw.set_eeg_reference(ref_channels='average',projection=False,verbose='WARNING')

    
    #滤波
    #raw.compute_psd().plot()
    #raw.plot()
    #watch(raw.get_data())
    raw=reform(denose2(raw.get_data()))
    #raw.compute_psd().plot()
    #raw.plot()
    #plt.show()
    #watch(raw.get_data())
    raw.filter(3.5,31,method='fir',fir_window='hamming',phase='zero-double',verbose='WARNING')
    #raw.compute_psd().plot()
    #raw.plot()
    #watch(raw.get_data())
    raw=reform(regular(raw.get_data()))
    #raw.compute_psd().plot()
    #raw.plot(scalings=0.0002)
    #plt.show()
    #watch(raw.get_data())
    raw.filter(3.5,31,method='fir',fir_window='hamming',phase='zero-double',verbose='WARNING')
    #raw.compute_psd().plot()
    #raw.plot(scalings=0.0002)
    #plt.show()
    #watch(raw.get_data())
    raw.resample(sfreq=128,window='hamming',verbose='WARNING')
    
    #if raw_path[7:9]=="31":
    #raw.plot(scalings=0.2)
    #plt.show()
    #print(raw.info)
    
    #时间截取
    if stage=="0":
        if raw_path[7:9]=="31":
            raw.crop(tmin=0,tmax=60,verbose='WARNING')
        else:
            raw.crop(tmin=60,tmax=120,verbose='WARNING')
    elif stage=="1":
        raw.crop(tmin=0,tmax=60,verbose='WARNING')
    #wavelet transform
    #a,f=pywt.cwt(raw.get_data()[0],scales=numpy.arange(1,40),wavelet="gaus1",sampling_period=128)
    #plt.imshow(a,cmap='PRGn',aspect='auto',vmax=abs(a).max(), vmin=-abs(a).max())  
    #plt.show()
    #ICA
    #ica=ICA(method='picard',fit_params=dict(ortho=False,extended=True),random_state=97,verbose='WARNING')                #infomax最好
    #ica.fit(raw,verbose='WARNING')
    
    
    #ica.exclude=[]
    #eog_exclude=[]
    #mascle_exclue=[]
    
    #eog_inds,eog_scores=ica.find_bads_eog(raw,['Fp1','Fp2'],measure='correlation',threshold=0.9,verbose='WARNING')
    #print(eog_inds)
    #eog_exclude=list(set(get_eog_exclude(eog_scores))|set(eog_inds))
    

    #mascle_inds,mascle_scores=ica.find_bads_muscle(raw,verbose='WARNING',threshold=0.9)
    #mascle_exclude=get_mascle_exclude(mascle_scores)
    
    
    #ica.exclude=list(set(eog_inds)|set(mascle_inds))
    
     
    #print("eog,mascle",eog_inds,mascle_inds)
    #ica.apply(raw,verbose='WARNING')
    
    #重参考
    #raw.set_eeg_reference(ref_channels='average',projection=False,verbose='WARNING')
    
    #breakpoint()
    #保存数据
    save_dict={}
    #save_dict['ch_names']=raw.info['ch_names']
    #save_dict['meas_date']=str(raw.info['meas_date'])
    #save_dict['sfreq']=raw.info['sfreq']
    #save_dict['reference']='average'
    #save_dict['pass']='4-45 hz'
    #save_dict['stage']=stage
    #save_dict['time_range']=str(t_start)+'-'+str(t_stop)+'s'
    #save_dict['channel_location']='standard_1020'
    save_dict['data']=raw.get_data()
    full_write_path=write_path+raw_path[7:9]+'-'+stage+'.mat'
    scipy.io.savemat(full_write_path,save_dict)

    
    #载出数据
    raw.close()
    return

if __name__=="__main__":

    
    list_dirs=os.listdir(read_path)
    for sub_path in list_dirs:
        
        if sub_path[10]=="1":
            stage="0"
        elif sub_path[10]=="2":
            stage="1"
        mnePreprocess(sub_path,stage)





        


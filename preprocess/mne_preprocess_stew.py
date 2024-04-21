
import numpy
#import pathlib
import mne
#from mne.preprocessing import ICA
import os
#import ast
import scipy.io
import matplotlib.pyplot as plt

write_path="./after/stew/"
read_path="./raw/stew/"
'''
def get_eog_exclude(scores):
    
    mean_exclude=numpy.argsort(numpy.mean(numpy.abs(scores),axis=0))[-4:]   
    #print(mean_exclude)

    max_exclude=numpy.argsort(numpy.abs(scores))[:,-2:].flatten()
    
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
        plt.title(idx+1)
        plt.plot(i)
        plt.show()
    plt.plot(numpy.stack(list(range(len(data.flatten()))))/len(data.flatten())*10,data.flatten())
    plt.show()
def reform(data):
    
    ch_names=['F3','F7','T7','P7','O1','O2','P8','T8','F8','F4']
    sfreq=128
    ch_types='eeg'
    info=mne.create_info(ch_names,sfreq,ch_types=ch_types,verbose='WARNING')
    raw=mne.io.RawArray(data,info,verbose='WARNING')
    #raw.set_montage('standard_1020',on_missing='raise',verbose='WARNING')
    return raw

def mnePreprocess(raw_path,stage):

    #读取
    full_read_path=read_path+raw_path
    samples=numpy.loadtxt(full_read_path)
    
    info=mne.create_info(['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4'],128,ch_types='eeg',verbose='WARNING')
    raw=mne.io.RawArray(samples.T,info,verbose='WARNING')
    
    print(raw_path)
    
    #载入数据
    raw.load_data(verbose='WARNING')
    raw.reorder_channels(['F3','F7','T7','P7','O1','O2','P8','T8','F8','F4'])

    #channels locaion
    #montage=mne.channels.make_standard_montage('standard_1020')
    #raw.set_montage('standard_1020',on_missing='raise',verbose='WARNING')

    #重参考
    #raw.set_eeg_reference(ref_channels='average',projection=False,verbose='WARNING')

    
    
    #滤波
    #raw.compute_psd().plot()
    #raw.plot()
    #plt.show()
    #breakpoint()
    #watch(raw.get_data())
    raw=reform(denose2(raw.get_data()))
    #raw.compute_psd().plot()
    #raw.plot(scalings=0.00002)
    #watch(raw.get_data())
    raw.filter(3.5,31,method='fir',fir_window='hamming',phase='zero-double',verbose='WARNING')
    #raw.compute_psd().plot()
    #raw.plot(scalings=10)
    #watch(raw.get_data())
    #plt.show()
    
    raw=reform(regular(raw.get_data()))
    #raw.compute_psd().plot()
    #raw.plot(scalings=1)
    #watch(raw.get_data())
    raw.filter(3.5,31,method='fir',fir_window='hamming',phase='zero-double',verbose='WARNING')
    #raw.compute_psd().plot()
    #raw.plot()
    #plt.show()
    #watch(raw.get_data())
    #breakpoint()

    #时间截取
    if stage=="0":
        raw.crop(tmin=45,tmax=105,verbose='WARNING')
    elif stage=="1":
        raw.crop(tmin=45,tmax=105,verbose='WARNING')
    #ICA
    #ica=ICA(method='picard',fit_params=dict(ortho=False,extended=True),random_state=97,max_iter=10000,verbose='WARNING')         
    #ica.fit(raw,verbose='WARNING')
    #ica.plot_components(inst=raw)

    #ica.exclude=[]
    #eog_exclude=[]
    #mascle_exclue=[]
    
    #eog_inds,eog_scores=ica.find_bads_eog(raw,['AF3','AF4'],measure='correlation',threshold=0.7,verbose='WARNING')
    
    #eog_exclude=list(set(get_eog_exclude(eog_scores))|set(eog_inds))
    

    #mascle_inds,mascle_scores=ica.find_bads_muscle(raw,verbose='WARNING',threshold=0.9)
    #mascle_exclude=get_mascle_exclude(mascle_scores)
    
    
    #ica.exclude=list(set(eog_inds)|set(mascle_inds))
    
        
    #print("eog,mascle",eog_inds,mascle_inds)
    #ica.apply(raw,verbose='WARNING')
    #raw.plot(scalings=50,title="3")
    #重参考
    #raw.set_eeg_reference(ref_channels=['F7','T7','P7','O1','O2','P8','T8','F8'],projection=False,verbose='WARNING')
    #raw.plot(scalings=50,title="4")
    #breakpoint()
    #保存数据
    save_dict={}
    #save_dict['ch_names']=raw.info['ch_names']
    #save_dict['sfreq']=raw.info['sfreq']
    #save_dict['reference']='average'
    #save_dict['pass']='4-45 hz'
    #save_dict['channel_location']='standard_1020'
    save_dict['data']=raw.get_data()
    full_write_path=write_path+raw_path[3:-7]+'-'+stage+'.mat'
    scipy.io.savemat(full_write_path,save_dict)

       
    #载出数据
    raw.close()
    return
if __name__=="__main__":
    
    list_dirs=os.listdir(read_path)
    for sub_path in list_dirs:

        if sub_path[-5]=="i":
            stage="1"
        elif sub_path[-5]=="o":
            stage="0"
        mnePreprocess(sub_path,stage)
    






    

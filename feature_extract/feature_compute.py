from scipy.stats import kurtosis,skew
import numpy

import math
import pywt
from pywt import wavedec
from mne.time_frequency import psd_array_welch as paw
import matplotlib.pyplot as plt

def compute_DE(signal):
    variance=numpy.var(signal,ddof=1)
    return -math.log(2*math.pi*math.e*(variance**2))/2

def compute_ZCR(data):
    times=0
    if data[0] >0:
        last=1
    elif data[0]<0:
        last=-1
    elif data[0]==0:
        last=0
        
    for i in range(len(data)-1):
        if data[i+1]>0:
            temp=1
        elif data[i+1]<0:
            temp=-1
        elif data[i+1]==0:
            temp=0
        if last!=0:
            if temp!=0:
                if temp!=last:
                    times+=1
                last=temp
        elif last==0:
            last=temp
            
    return (times/(len(data)-1))

def diff(x,n_step):
    n=len(x)
    s=0
    for i in range(n):
        if i+n_step<n:
            s+=(x[i]-x[i+n_step])    
    s=s/(n-n_step)
    return s

def waveEn(data):
    
    #小波能量
    en_list=[]
    list_dec=wavedec(data,'db4')
    print(len(list_dec[3]))
    for decs in (list_dec[1:]):
        en=sum(decs**2)
        en_list.append(en)
    '''
    ##小波熵
    p_list=en_list/sum(en_list)

    logp_list=[]
    for p in p_list:
        logp=math.log(p)
        logp_list.append(logp)
    lee=(sum(p_list*logp_list))*(-1)
    '''
    return sum(en_list)

    

    
    
    

def get_feature(data):

        
    
    #求特征
    
    #F_mean=numpy.mean(data)                                    #均值 0.55
    #F_abs_mean=numpy.mean(abs(data))                           #绝对均值 0.5
    #F_peak=max(abs(data))                                      #峰值 0.52
    #F_max=max(data)                                            #最大值 0.55
    #F_min=min(data)                                            #最小值0.47
    #F_var=numpy.var(data,ddof=1)                               #方差0.48
    #F_abs_var=numpy.var(abs(data),ddof=1)                      #绝对方差0.5
    #F_std=numpy.std(data,ddof=1)                               #标准差 0.48
    #F_abs_std=numpy.std(abs(data),ddof=1)                      #绝对标准差 0.5
    #F_rms=math.sqrt(numpy.mean(data**2))                       #均方根 0.48
    #F_sf=F_rms/F_abs_mean                                      #波形因子 0.4
    #F_skewness=skew(data)                                      #偏斜度 0.49
    #F_kurtosis=kurtosis(data)                                  #峰度 0.4
    #F_cf=F_peak/F_rms                                          #波峰因子 0.5
    #F_pi=F_peak/F_abs_mean                                     #脉冲指数 0.49
    #F_ds1=diff(data,1)                                         #一阶差分 0.49
    #F_ds2=diff(data,2)                                         #二阶差分 0.5
    #F_ZCR=compute_ZCR(data)                                    #过零率 0.49
    #F_DE=compute_DE(data)                                      #微分熵
    #F_EN=waveEn(data)                                        #对数能量熵
    #kk
    
    #psd
    
    p,f=paw(data,sfreq=128,fmin=4,fmax=30,average='mean',n_per_seg=128,n_overlap=64,n_fft=512,verbose='warning')        #sfreq
    #p=10*numpy.log10(p)
    #plt.plot(f,10*numpy.log10(p))
    #p,f=paw(data,sfreq=128,fmin=4,fmax=20,average='mean',n_per_seg=128,n_overlap=0,n_fft=4096)#,verbose='warning')        #sfreq
    #print(len(f),len(p))
    #plt.plot(f,10*numpy.log10(p),color='blue')
    #plt.show()
    #breakpoint()
    #print(p.shape)
    #psd_theta=numpy.mean(p[(f>=4)&(f<8)])
    #psd_alpha=numpy.mean(p[(f>=8)&(f<12)])
    #psd_beta=numpy.mean(p[(f>=12)&(f<30)])
    #psd_gamma=numpy.mean(p[(f>=30)&(f<40)])
    
    psd=p[(f>=4)&(f<12)]
    
    #feature_list=[psd_theta,psd_alpha]
    
    return psd  
    

def dataset_feature(data):
    
    output=[]
    for seg in data:
        c=[]
        for chs in seg:
            c.append(get_feature(numpy.array(chs)))
        output.append(c)
    output=numpy.array(output)#,dtype=numpy.float32)
    
    return output



if __name__=="__main__":
    
    print()
    
    
    
   
            

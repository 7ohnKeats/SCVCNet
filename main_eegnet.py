import tensorflow as tf
import numpy
import random


from tensorflow.keras.callbacks import ModelCheckpoint
from model.EEGModels import EEGNet
from get_data_data import eegmat,stew,nback,eegmat_test,stew_test,nback_test,eegmat_val,stew_val,nback_val
from sklearn.metrics import accuracy_score,f1_score

numpy.random.seed(42)
tf.random.set_seed(42)
random.seed(42)



class Main:
    def __init__(self,train=None,valid=None,test=None):

        self.train=train
        self.valid=valid
        self.test=test
            
    def select_data(self,data):
        d=[]
        l=[]
        c=[]
        for i,j,k in data:
            i=i.reshape(i.shape+(1,))
            j=tf.one_hot(j,2)
            d.append(i)
            l.append(j)
            c.append(k)   
        d=numpy.stack(d)
        l=numpy.stack(l)
        c=numpy.stack(c)
        
        return d,l,c
    def cut_sub(self,x,y,c):
        
        sub=set(c.tolist())
        num_sub=len(sub)
        num_sample=len(c)
        sps=num_sample//num_sub
        
        xx=x.reshape(num_sub,sps,x.shape[-3],x.shape[-2],x.shape[-1])
        yy=y.reshape(num_sub,sps,y.shape[-1])
        cc=c.reshape(num_sub,sps)
        
        return xx,yy,cc
    
    def score(self,labels,probs):
        preds=probs.argmax(axis=-1)
        labels=labels.argmax(axis=-1)
        acc=accuracy_score(y_true=labels,y_pred=preds)
        f1=f1_score(y_true=labels,y_pred=preds,average="macro")
        #print(acc,f1)
        return acc,f1
    
    def clf(self,d1,t1,c1,d2,t2,c2,d3,t3,c3):
        eegnet=EEGNet(nb_classes=2,Chans=10,Samples=2560,dropoutRate=0.25,norm_rate=0.25)
        opt=tf.keras.optimizers.Adam(learning_rate=0.0001)
        crt=tf.keras.losses.CategoricalCrossentropy()
        inp=tf.data.Dataset.from_tensor_slices((d1,t1))
        inp=inp.shuffle(buffer_size=1024).batch(10)
        for i in range(5):
            for d,t in inp:
                with tf.GradientTape() as tape:
                    p=eegnet(d,training=True)
                    loss=crt(t,p)
                grads=tape.gradient(loss,eegnet.trainable_weights)
                opt.apply_gradients(zip(grads,eegnet.trainable_weights))
            print(i,loss)
        #eegnet.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        #eegnet.fit(d1,t1,batch_size=10,epochs=10,verbose=2)

        #d3,t3,c3=self.cut_sub(d3,t3,c3)
        #d2,t2,c2=self.cut_sub(d2,t2,c2)
        p1=eegnet.predict(d1,verbose=3)
        #acc=eegnet.evaluate(d1,t1,verbose=2)
        acc,f1=self.score(t1,p1)
        print(acc,f1)
        print()
        p2=eegnet.predict(d2,verbose=3)
        p3=eegnet.predict(d3,verbose=3)
        acc2,f12=self.score(t2,p2)
        acc3,f13=self.score(t3,p3)
        print(acc2,'\t',f12)
        print(acc3,'\t',f13)
        
        '''
        for idx,(d,t,c) in enumerate(zip(d2,t2,c2)):
            p=eegnet.predict(d,verbose=3)
            acc,f1=self.score(t,p)
            print(acc,"\t",f1)

        print()
        for idx,(d,t,c) in enumerate(zip(d3,t3,c3)):
            p=eegnet.predict(d,verbose=3)
            acc,f1=self.score(t,p)
            print(acc,"\t",f1)
        '''
        return
    
    def forward(self):
        
        d1,t1,c1=self.select_data(self.train)
        d2,t2,c2=self.select_data(self.valid)
        d3,t3,c3=self.select_data(self.test)
        
        self.clf(d1,t1,c1,d2,t2,c2,d3,t3,c3)
        return
    
if __name__=='__main__':
    
    #print(len(nback),len(eegmat),len(stew))
    main=Main(train=stew,valid=nback_val,test=eegmat)
    main.forward()
    print("a")
   
    
        
    
    

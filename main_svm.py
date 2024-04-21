import numpy
import random

from sklearn import svm
from get_data_feat import eegmat,stew,nback,eegmat_test,stew_test,nback_test,eegmat_val,stew_val,nback_val
from sklearn.metrics import accuracy_score,f1_score

numpy.random.seed(42)
            
class Main:
    def __init__(self,train,valid,test):

        self.train=train
        self.valid=valid
        self.test=test    
        
    
    def select_data(self,data):
        d=[]
        l=[]
        for i,j,_ in data:
            d.append(i)
            l.append(j)
            
        d=numpy.stack(d)
        l=numpy.stack(l)
        return d,l

    def score(self,probs,labels):
    
        acc=accuracy_score(y_true=labels,y_pred=probs)
        f1=f1_score(y_true=labels,y_pred=probs,average="macro")
        print(acc,'\t',f1)
        return acc
    
    def svm(self,x1,y1,x2,y2,x3,y3):
        
        svc=svm.SVC(max_iter=10000,random_state=42,C=0.3,kernel='sigmoid',gamma=600,degree=2)
        svc.fit(x1,y1)
        
        p1=svc.predict(x1)
        p2=svc.predict(x2)
        p3=svc.predict(x3)

        self.score(p1,y1)
        self.score(p2,y2)
        self.score(p3,y3)

        return  
    
    def forward(self):
        
        x1,y1=self.select_data(self.train)
        x2,y2=self.select_data(self.valid)
        x3,y3=self.select_data(self.test)
        
        x1=x1.reshape(len(x1),-1)
        x2=x2.reshape(len(x2),-1)
        x3=x3.reshape(len(x3),-1)
        
        self.svm(x1,y1,x2,y2,x3,y3)
        return
    
if __name__=='__main__':
    
    
    main=Main(train=nback,valid=stew_val,test=eegmat)
    main.forward()
    print()
   
    
        
    
    

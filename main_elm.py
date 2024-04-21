import numpy
import random
import hpelm
from sklearn.metrics import accuracy_score,f1_score
from get_data_feat import eegmat,stew,nback,eegmat_test,stew_test,nback_test,eegmat_val,stew_val,nback_val


numpy.random.seed(42)

        
def prob2label(prob):
    out=numpy.zeros(len(prob))
    
    for idx in range(len(prob)):
        if prob[idx]>0.5:
            out[idx]=1
    return out

class ELM(hpelm.elm.ELM):
        
    def score(self,x,t):
        pp=self.predict(x)
        p=prob2label(pp)
        acc=accuracy_score(y_true=t,y_pred=p)
        f1=f1_score(y_true=t,y_pred=p,average="macro")
        print(acc,'\t',f1)
        return 
        
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

    def elm(self,x1,y1,x2,y2,x3,y3):
        elm=ELM(inputs=320,outputs=1,norm=0.01)
        elm.add_neurons(number=600,func='tanh')
        
        err=elm.train(x1,y1)
        
        #elm.score(x1,y1)
        elm.score(x2,y2)
        elm.score(x3,y3)
        return

    
    def forward(self):
        
        x1,y1=self.select_data(self.train)
        x2,y2=self.select_data(self.valid)
        x3,y3=self.select_data(self.test)
        
        x1=x1.reshape(len(x1),-1)
        x2=x2.reshape(len(x2),-1)
        x3=x3.reshape(len(x3),-1)
        
        self.elm(x1,y1,x2,y2,x3,y3)
        return
    


if __name__=='__main__':
    
    
    main=Main(nback,stew_val,eegmat)
    
    main.forward()
    print()
   
    
        
    
    

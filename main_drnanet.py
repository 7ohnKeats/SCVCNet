import torch
import numpy
from sklearn.metrics import accuracy_score,f1_score
from model.model_with_self_attention import GRUModel

from get_data_feat_norm import eegmat,stew,nback,eegmat_test,stew_test,nback_test,eegmat_val,stew_val,nback_val

torch.manual_seed(42)
numpy.random.seed(42)

eegmat=torch.utils.data.DataLoader(eegmat,batch_size=10,shuffle=True)
eegmat_val=torch.utils.data.DataLoader(eegmat_val,batch_size=10,shuffle=True)
eegmat_test=torch.utils.data.DataLoader(eegmat_test,batch_size=10,shuffle=True)
stew=torch.utils.data.DataLoader(stew,batch_size=10,shuffle=True)
stew_val=torch.utils.data.DataLoader(stew_val,batch_size=10,shuffle=True)
stew_test=torch.utils.data.DataLoader(stew_test,batch_size=10,shuffle=True)
nback=torch.utils.data.DataLoader(nback,batch_size=10,shuffle=True)
nback_val=torch.utils.data.DataLoader(nback_val,batch_size=10,shuffle=True)
nback_test=torch.utils.data.DataLoader(nback_test,batch_size=10,shuffle=True)

model=GRUModel(input_size=32,hidden_size=128,num_layers=2,num_classes=2)
crt=torch.nn.CrossEntropyLoss()
optim=torch.optim.Adam(model.parameters(),lr=0.001)


for epoch in range(2):
    model.train()
    losses=[]
    for idx,(d,l,_) in enumerate(nback):
        optim.zero_grad()
        d=d.to(dtype=torch.float)
        p,_=model(d)
        loss=crt(p,l)
        loss.backward()
        optim.step()
        losses.append(loss.item())
    losses=sum(losses)
    print(epoch,losses)
        

model.eval()
with torch.no_grad():
    pps=[]
    ls=[]
    for d,l,_ in stew_test:
        d=d.to(dtype=torch.float)
        p,_=model(d)
        pp=p.argmax(axis=-1)
        pps.extend(pp)
        ls.extend(l)
    acc=accuracy_score(ls,pps)
    f1=f1_score(ls,pps,average="macro")
    print(acc,'\t',f1)

model.eval()
with torch.no_grad():
    pps=[]
    ls=[]
    for d,l,_ in eegmat:
        d=d.to(dtype=torch.float)
        p,_=model(d)
        pp=p.argmax(axis=-1)
        pps.extend(pp)
        ls.extend(l)
    acc=accuracy_score(ls,pps)
    f1=f1_score(ls,pps,average="macro")
    print(acc,'\t',f1)
        




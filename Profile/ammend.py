import pickle as p

with open('ProfileResult.pkl','rb') as f:
    d=p.load(f)

 

dd={k1:{k2:v2 for k2,v2 in v1.items() if v2['Latency']!=20000} for k1,v1 in d.items()}

with open('R.pkl','wb') as f:
    p.dump(dd,f)



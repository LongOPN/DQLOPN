"""Draft, Version 0: LOPN divide and conquer"""
import os
import math
import itertools
import argparse
import time
import random
import torchvision.utils as tor
import pandas as pd
import numpy as np
import torch
import torch.nn as nn 
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torch.optim as optim
from models.modellopn import LOPN
from models.c3d import C3D
from models.r3d import R3DNet
from models.r21d import R2Plus1DNet
from PIL import Image
from shutil import copyfile
from torchvision import transforms
import cv2
import numpy
import matplotlib.pyplot as plt
from datasets.ucf101 import UCF101FOPDataset
 
 
  
"""Compare using 4-tuple model"""
def compare( tuple_clips, tuple_len, ckpt,device, bsmodel= 'r3d' ,gpu='0'):
    len=tuple_len
    if bsmodel == 'c3d':
        base = C3D(with_classifier=False)
    elif bsmodel == 'r3d':
        base = R3DNet(layer_sizes=(1,1,1,1), with_classifier=False)
    elif bsmodel == 'AlexNet':
        base = AlexNet(with_classifier=False, return_conv=False)
    elif bsmodel == 'r21d':   
        base = R2Plus1DNet(layer_sizes=(1,1,1,1), with_classifier=False)
    opn = LOPN(base_network=base, feature_size=256, tuple_len=len ).to(device)   
    opn.load_state_dict(torch.load(ckpt))
 
    # Force the pytorch to create context on the specific device 
    model=opn
    torch.set_grad_enabled(False) 
    model.eval()
    pts=[]  
    outputs=[]
 
    inputs = torch.tensor(tuple_clips).to(device)
    # forward  
    if torch.cuda.is_available():
      model.to(device)
    #torch.set_grad_enabled(False)
    inputs = inputs.float()
    outputs = model(inputs)
    #print('me1')
    pts = torch.argmax(outputs, dim=1)
    return pts
 

"""SortFW"""
def sortmefw(classes,pts,prob4,mapp4 ):
    map4=[]
    sortp4=[]
    h=classes[pts]    
    for i in range(4):
        #fwi=fwcls[i]
        pi=h.index(i)
        ho=prob4[0,pi,:,:,:]
        sortp4.append( ho) 
        ma=mapp4[pi]
        map4.append( ma) 
    sortp4=torch.stack(sortp4)
    return sortp4, map4
 

"""SortBW"""
def sortmebk(classes,pts,prob4,mapp4 ):
    map4=[]
    sortp4=[]
    h=classes[23-pts]    
    for i in range(4):
        pi= h.index(i)
        ho=prob4[0,pi,:,:,:]
        sortp4.append( ho) 
        ma=mapp4[pi]
        map4.append( ma) 
    sortp4=torch.stack(sortp4)
    return sortp4,map4
 
 
 
"""FlagCheck"""
def flgcheck(classes,pts):
    #Mask
    a=[0,1,5,6]   
    sortp3rev=[]
    sortp4=[]
    cl=classes[pts]
    for i in range(4):
        cli=cl.index(i)
        ho=a[cli]
        sortp4.append( ho) 
    sortp4f=sortp4
 
    if sortp4f.index(0)>sortp4f.index(1):
      flg1=1
    else:
      flg1=0
    return flg1
 
 


# manage repeat flag in opimal mode (flmap=mask) 
def flgcheck3p(classes,pts,a,flmap):
    sortp3rev=[]
    sortp4=[]
    cl=classes[pts]
    for i in range(4):
        #cli=cl[i]
        cli=cl.index(i)
        ho=a[cli]
        sortp4.append( ho) 
    sortp4f=sortp4
    if sortp4f.index(flmap[0])>sortp4f.index(flmap[1]):
      flg1=1
    else:
      flg1=0
  
    return flg1


 
def revme( sortp3,  map3=[0,1,2,3],size=4):
    rev=[]
    map3n=[]
    for i in range(size-1, -1,-1):
      pl=sortp3[i,:,:,:]
      m3=map3[i]
      map3n.append(m3) 
      rev.append(pl) 
    sortp33=torch.stack(rev)
    return sortp33 
 
 
def checkrev( sortp3,  map3,flg3):
    if flg3==1:
      sortp3rev=[]
      maprev3=[]
      for i in range(4-1, -1,-1):
          ho=sortp3[i,:,:,:]
          sortp3rev.append( ho ) 
          ma=map3[i]
          maprev3.append( ma) 
      sortp3=torch.stack(sortp3rev)
      map3=maprev3
    return sortp3,map3
 

# append two tuple 
def twoappend(newsort,newmap, sortp2, map2):
  for i in  range(len(map2)) :  
      if map2[i] != -1:
        newmap.append(map2[i])
        newsort.append(sortp2[i,:,:,:]  )
  return newsort , newmap
 
# append frames 
def selectframe(mapparent1,sortparent1,subp1,submap1,lep1):
    cnt1=0
    fp1=0
    flgmp1=0
    #print(mapparent1)
    #print('lep',lep1)
    while (fp1<lep1):
       
      if (mapparent1[cnt1:5]==[]) :
        flgmp1=1
        #print('flgm',flgmp1)
        break
      else:
          if mapparent1[cnt1] !=-1  :
            subp1.append(sortparent1[cnt1, :, :, :])
            submap1.append(mapparent1[cnt1])
            fp1+=1;
            cnt1+=1
          else: 
            cnt1+=1
      #print('cnt',cnt1)
    return subp1,submap1,flgmp1
 
#############################my main
bsmodel='r21d' 
# model dir here 4-tuple
ckpt='/content/drive/My Drive/opnmodelR21D/vcopopntl4R21D/r21d_cl16_it8_tl4_05081022/best_model_158_0.499__0.860.pt'
gpu=0
torch.backends.cudnn.benchmark = True
# Force the pytorch to create context on the specific device 
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
print("device",device)
 
 
gpu=gpu
seed=632
if seed:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if gpu:
        torch.cuda.manual_seed_all(seed)
 
 
 
m1=1
classes = list(itertools.permutations(list(range(4))))
# input from dataloader 
#test_dataset = UCF101FOPDataset('data/ucf101/ucf32',  8, 8, False, test_transforms)
 
#test_dataloader = DataLoader(test_dataset, 1, shuffle=False, num_workers=0, pin_memory=True)                             
all=0
acc=0
#for i, data in  enumerate(test_dataloader, 1):
print('sampleeeeeeeeeeeeeeeeeeeeee')
pts=[]
targets=[]
outputs=[]
 
 #sample input 
 
 
a= cv2.imread('/content/drive/My Drive/VCOP/res8/a225image_5tpl8.jpg')
b= cv2.imread('/content/drive/My Drive/VCOP/res8/a225image_6tpl8.jpg')
c= cv2.imread('/content/drive/My Drive/VCOP/res8/a225image_4tpl8.jpg')
d= cv2.imread('/content/drive/My Drive/VCOP/res8/a225image_7tpl8.jpg')
e= cv2.imread('/content/drive/My Drive/VCOP/res8/a225image_0tpl8.jpg')
f= cv2.imread('/content/drive/My Drive/VCOP/res8/a225image_1tpl8.jpg')
g= cv2.imread('/content/drive/My Drive/VCOP/res8/a225image_3tpl8.jpg')
h= cv2.imread('/content/drive/My Drive/VCOP/res8/a225image_2tpl8.jpg')
 
# cnd svae counter
#optim unpotimized=1                optim>1   optimized some bug.
optim=1
cndt=9
 
  
 
test_transforms = transforms.Compose([
    #transforms.Resize((128, 171)),
    #transforms.CenterCrop(112),
    #transforms.RandomCrop(20,30),
    transforms.ToTensor()
])
 
toPIL = transforms.ToPILImage()
a= cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
b= cv2.cvtColor(b, cv2.COLOR_BGR2RGB) 
c= cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
d= cv2.cvtColor(d, cv2.COLOR_BGR2RGB)
e= cv2.cvtColor(e, cv2.COLOR_BGR2RGB)
f= cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
g= cv2.cvtColor(g, cv2.COLOR_BGR2RGB)
h= cv2.cvtColor(h, cv2.COLOR_BGR2RGB)
 
 
 
a = toPIL(a) # PIL image
a = test_transforms(a) # tensor [C x H x W]
 
b = toPIL(b) # PIL image
b = test_transforms(b) # tensor [C x H x W]
 
c = toPIL(c) # PIL image
c = test_transforms(c) # tensor [C x H x W]a = toPIL(a) # PIL image
 
 
d = toPIL(d) # PIL image
d = test_transforms(d) # tensor [C x H x W]a = toPIL(a) # PIL image
 
e = toPIL(e) # PIL image
e = test_transforms(e) # tensor [C x H x W]a = toPIL(a) # PIL image
 
f = toPIL(f) # PIL image
f = test_transforms(f) # tensor [C x H x W]a = toPIL(a) # PIL image
 
g = toPIL(g) # PIL image
g = test_transforms(g) # tensor [C x H x W]a = toPIL(a) # PIL image
 
h = toPIL(h) # PIL image
h = test_transforms(h) # tensor [C x H x W]a = toPIL(a) # PIL image
 
 
 
 
 
order1=[0,1,2,3]
order2=[0,1,4,5]
order3=[0,1,6,7]  
 
mapp1=[0,1,2,3]
mapp2=[0,1,4,5]
mapp3=[0,1,6,7]  
 
problem1=[]
problem1.append(a)
problem1.append(b)
problem1.append(c)
problem1.append(d)
 
 
problem2=[]
problem2.append(a)
problem2.append(b)
problem2.append(e)
problem2.append(f)
 
problem3=[]
problem3.append(a)
problem3.append(b)
problem3.append(g)
problem3.append(h)
 
 
prob1= torch.stack(problem1)
prob2= torch.stack(problem2)
prob3= torch.stack(problem3)
#---------------------------------tuple 1 and 2 and 3  created
cntt=0
#---------------------------------Visualize inpu tuple   
for i in range(4):
  plt.imshow(prob1[i,:,:,:].permute(1,2,0))
  plt.show()
  tor.save_image(prob1[i,:,:,:], "restst/inp{}image_{}tpl8.jpg".format(str(cndt),str(cntt)) )  
  cntt+=1
print('prob2 0 1  4 5')
for i in range(2,4):
  plt.imshow(prob2[i,:,:,:].permute(1,2,0))
  plt.show()
  tor.save_image(prob2[i,:,:,:], "restst/inp{}image_{}tpl8.jpg".format(str(cndt),str(cntt)) )  
  cntt+=1
 
print('prob2 0 1  6 7')
for i in range(2,4):
  plt.imshow(prob3[i,:,:,:].permute(1,2,0))
  plt.show()
  tor.save_image(prob3[i,:,:,:], "restst/inp{}image_{}tpl8.jpg".format(str(cndt),str(cntt)) )  
  cntt+=1
 
 
prob1= prob1[None,:,:,:,:]
#print('shp',prob1.shape)
prob2= prob2[None,:,:,:,:]
#print('shp',prob1.shape)
prob3= prob3[None,:,:,:,:]
 
############### level 0  :Sort  each 3 sub arry-4      4     4    4
print('--------------------------Tuple 1---------------------------------------------')
mycnt=0
flgmap=mapp1   #keep true  directionkey
for myrep in range(optim):
        # Shuffle by random using pts
        pts1= compare( prob1   , 4, ckpt,device, bsmodel ,gpu='0') 
        # shuffle by random using random sort result.
        if myrep>=2 and ptstst!=0 and tstbef==ptstst and mycnt%3==0:
            pts1= np.random.randint(1, 11)
        mycnt+=1
        
        # Sort each tuple using pts ( comapres results ) 
        #sortp1, map1=  sortme(classes,pts1,prob1,mapp1 ) 
        flg1 = flgcheck3p(classes, pts1 ,mapp1,flgmap) 
        #sortp1,map1=  checkrev( sortp1 , map1,flg1)  
        #print('part1 map1 ',mapp1)
        if flg1==1:
            sortp1, map1= sortmebk(classes,pts1,prob1,mapp1)
        else:
            sortp1, map1= sortmefw(classes,pts1,prob1,mapp1 )

        tst=sortp1[None,:,:,:,:]
        ptstst1= compare( tst  , 4, ckpt,device, bsmodel  ,gpu='0') 
        ptstst=ptstst1

        if int(ptstst==0):
            break 
        else:
            tstbef=pts1
            prob1=sortp1[None,:,:,:,:]
            mapp1=map1  
      

 
################################# Tuple 2
print('--------------------------Tuple 2---------------------------------------------')
flgmap=mapp2    #keep true  directionkey
for myrep in range(optim):
      pts2= compare( prob2 , 4, ckpt,device, bsmodel  ,gpu='0') 
    
      flg2 = flgcheck3p(classes, pts2 ,mapp2,flgmap) 
      
 
      if flg2==1:
          sortp2, map2= sortmebk(classes,pts2,prob2,mapp2)
      else:
          sortp2, map2= sortmefw(classes,pts2,prob2,mapp2 )
      
      tst=sortp2[None,:,:,:,:]
      ptstst1= compare( tst  , 4, ckpt,device, bsmodel  ,gpu='0') 
      ptstst=ptstst1 

      if int(ptstst==0):
          break 
      else:
          tstbef=pts2
          prob2=sortp2[None,:,:,:,:]
          mapp2=map2  
    

################################# Tuple 3
print('--------------------------Tuple 3---------------------------------------------')
mycnt=0
flgmap=mapp3   #keep true  directionkey
for myrep in range(optim):
    # Shuffle by random using pts
    pts3= compare( prob3   , 4, ckpt,device, bsmodel ,gpu='0') 
    if myrep>=2 and ptstst!=0 and   mycnt%3==0:
        pts3= np.random.randint(1, 11)
    mycnt+=1
    
    flg3 = flgcheck3p(classes, pts1 ,mapp3,flgmap) 
 
    if flg3==1:
        sortp3, map3= sortmebk(classes,pts3,prob3,mapp3)
    else:
        sortp3, map3= sortmefw(classes,pts3,prob3,mapp3 )
 
    tst=sortp3[None,:,:,:,:]
    ptstst= compare( tst  , 4, ckpt,device, bsmodel  ,gpu='0') 
  
    if int(ptstst==0):
        break 
    else:
        tstbef=pts3
        prob3=sortp3[None,:,:,:,:]
        mapp3=map3  
    
 
###################################### merge tuple 2 and 3
############### level 1-1  : merge 2 second arrays-2    4   4-2   4-2 = 4  4  
print('-------------------------- end all 3 Tuples---------------------------------------------') 
a= map2.index(0) 
map2[a]=-1
a= map2.index(1) 
map2[a]=-1
#print('map3',map3)
a= map3.index(0) 
map3[a]=-1
a= map3.index(1) 
map3[a]=-1
 
print('-------------------------- Merge stage Tuples 2 and 3---------------------------------------------') 
 
############### level 1-2 Sort 2 2 merged array  :first 2 2 check     check 1arry-   4      sort 4
resultsort=[]
resultmap=[]  
newmap=[]
newsort=[] 
 
newsort , newmap = twoappend(newsort,newmap, sortp2, map2)
newsort , newmap = twoappend(newsort,newmap, sortp3, map3)
#print('map 2 map3 aded ' ,map2,map3,newmap)
sort4=torch.stack(newsort )
print('-------------------------- Merge stage Tuples 2 and 3 Sorting---------------------------------------------') 
mapflg=newmap
for myrep  in range(optim):
  sort4=sort4[None,:,:,:,:] 
  pts4= compare( sort4 , 4, ckpt,device, bsmodel ,gpu='0') 
  #sortp4, map4= sortme(classes,pts4,sort4,newmap )
  #flg4= flgcheck(classes, pts4 )
  flg4 = flgcheck3p(classes, pts4 ,newmap,mapflg) 
  #sortp4,map4=  checkrev( sortp4 , map4,flg4) 
 
  if flg4==1:
      sortp4, map5= sortmebk(classes,pts4,sort4,newmap )
  else:
      sortp4, map5= sortmefw(classes,pts4,sort4,newmap )
  #print('map 2 map3 aded ' ,map2,map3,newmap,'flg newsor',flg4,map5)
  
  tst=sortp4[None,:,:,:,:]
  ptstst= compare( tst  , 4, ckpt,device, bsmodel  ,gpu='0') 
  print('myreeeeep',myrep,'ptst',ptstst,'bef',pts4,'rl flg',flg4,pts4)

  if int(ptstst==0):
      break 
  else:
      tstbef=pts4
      sort4=sortp4
      newmap=map5  
  
 
print('-------------------------- end merge Tuples 1 and 2---------------------------------------------') 
 
#print(sortp1.shape,sortp4.shape)
#print('map1',map1,'map4',map5)
mapart1=map1
mapart2=map5
#-----------------------------parent 1  and parent2
mapparent1= mapart1
mapparent2= mapart2
sortparent1= sortp1
sortparent2= sortp4
############### level 2  sort  merged 1          start iteration
#-----------------------------Merge
while (1) :
    subp1=[]
    submap1=[]
    flgmp1=0
    flgmp2=0
    frm1=0
    frm2=0
    lep1=0
    lep2=0
    lp1=0
    lp2=0
    lep1=mapparent1.count(-1)
    lep2= mapparent2.count(-1)
    
    lp1=4- lep1
    lp2=4- lep2
 
    #if list 1 ended
    if (lp1>= 2 and lp2>=  2  and lp1+lp2>4):
        print('if 0-----------------------------------------------------')
        frm1=2
        frm2=2
        print('lp1>=2 and lp2>= 2 ')
 
    elif (lp1+lp2== 4):
          if (lp2> lp1):
            print('if 1 -------------------------------------------------')
            tmp=mapparent2
            mapparent2=mapparent1
            mapparent1=tmp
 
            tmp=sortparent2
            sortparent2=sortparent1
            sortparent1=tmp
            frm1=lp2
            frm2=lp1 
 
          else:
              frm1=lp1
              frm2=lp2
          
 
    elif  (lp1==0  ):
          print('if 2-----------------------------------------')
          frm2=lp2
          subp1,submap1,flgmp2=  selectframe(mapparent2,sortparent2,subp1,submap1,frm2)
          subp1=torch.stack(subp1)
          subp1=subp1[None,:,:,:,:]
          pts4= compare( subp1 , 4, ckpt,device, bsmodel  ,gpu='0') 
          #sortp4, map4= sortme(classes,pts4,subp1,submap1 )
          flg4= flgcheck(classes, pts4 )
          if flg4==1:
              sortp4, map4= sortmebk(classes,pts4,subp1,submap1 )
          else:
              sortp4, map4= sortmefw(classes,pts4,subp1,submap1 )
 
          for i in range(lp2 ):
            resultsort.append(sortp4[i, :, :, :] )
            resultmap.append(map4[i] )
          break
    #if list 2 ended
    elif  (lp2==0 ):
        print('if 3-------------------------------')
        frm1=lp1
        subp1,submap1,flgmp1=  selectframe(mapparent1,sortparent1,subp1,submap1,frm1)
        subp1=torch.stack(subp1)
        subp1=subp1[None,:,:,:,:]
        #print('shp',subp1.shape)
        pts4= compare( subp1  , 4, ckpt,device, bsmodel  ,gpu='0') 
        #sortp4, map4= sortme(classes,pts4,subp1,submap1 )
        flg4= flgcheck(classes, pts4 )
          
  
        if flg4==1:
            sortp4, map4= sortmebk(classes,pts4,subp1,submap1 )
        else:
            sortp4, map4= sortmefw(classes,pts4,subp1,submap1 )
        print(map4)
          
        for i in range(lp1 ):
          resultsort.append(sortp4[i, :, :, :] )
          resultmap.append(map4[i] )
        #resultsort,resultmap,flgmp1=  selectframe(sortp4,map4,resultsort,resultmap,frm1)
        break
 
  
    elif (lp1+lp2< 4):
        print('if 4----------------------------------------')
        print("exepition")
        exp=1
        break 
    print('===========================================================================================')
    #print('frm1 frm2',frm1,frm2)
    subp1=[]
    submap1=[]
    subp1,submap1,flgmp1=  selectframe(mapparent1,sortparent1,subp1,submap1,frm1)   
    #print('shp1',len(subp1)) 
    subp1,submap1,flgmp2=  selectframe(mapparent2,sortparent2,subp1,submap1,frm2)
  
  
    subp1=torch.stack(subp1)
    mapchild=submap1
 
    sortchild=0
  
    #optim=1
    flgmap=submap1
    mycnt=0
    for myrep in range(optim):
        subp1=subp1[None,:,:,:,:]
        # Shuffle by random using pts
        pts4= compare( subp1  , 4, ckpt,device, bsmodel  ,gpu='0') 
        if myrep>=2 and ptstst!=0 and   mycnt%3==0:
            pts4= np.random.randint(1, 11)
        mycnt+=1

   
       
        flg4= flgcheck3p(classes, pts4 ,submap1,flgmap) 
        #print('bef pts',pts4)
        if flg4==1:
            sortp4, map4= sortmebk(classes,pts4,subp1,submap1 )
        else:
            sortp4, map4= sortmefw(classes,pts4,subp1,submap1 )
        tst=sortp4[None,:,:,:,:]
         
        ptstst= compare( tst  , 4, ckpt,device, bsmodel  ,gpu='0') 
 
        if int(ptstst==0):
           break 
        else:
          tstbef=pts4
          subp1=sortp4
          submap1=map4
  
 
 
    #############level 4 check and- create sorted result
 
    print('lp1+lp2<',lp1+lp2,lp1,lp2)
    if lp1+lp2<= 4: 
      print('if  end----------------------------')
      #print('last ep',sortp4.shape)
      for i in range(lp1+lp2):
        resultsort.append(sortp4[i, :, :, :] )
        resultmap.append(map4[i] )
      break 
    else: 
      #print(sortp4.shape)
      resultsort.append(sortp4[0,:,:,:])
      resultsort.append(sortp4[1,:,:,:])
      resultmap.append(map4[0])
      resultmap.append(map4[1])
 
 
    checklist=[]
    p1=0
    p2=0
    for i in range(len(map4)):
      e=map4[i]    
      if map4[i] in mapparent1 :  
        a= mapparent1.index(e)
        checklist.append(1)
        p1+=1
        #print(a)
      if map4[i] in mapparent2 :  
        #print(i,map4[i]) 
        b= mapparent2.index(e)
        checklist.append(2)
        p2+=1
        #print(b)
    print(resultmap)
    for i in range(len(resultmap)):
      e=resultmap[i]    
      if resultmap[i]  in mapparent1 :  
        b=mapparent1.index(e) 
        mapparent1[b]=-1
      if resultmap[i]  in mapparent2 :  
        b=mapparent2.index(e)
        mapparent2[b]=-1
   
 
resultsort=torch.stack(resultsort) 
print('resultmap') 
for i in range(len(resultmap)):
  plt.imshow(resultsort[i,:,:,:].permute(1,2,0))
  plt.show()
  tor.save_image(resultsort[i,:,:,:], "restst/out{}image_{}tpl8.jpg".format(str(cndt),str(cntt)) )  
  cntt+=1
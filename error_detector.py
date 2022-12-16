#classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
import torch
import torch.backends.cudnn as cudnn
from robustbench.data import load_cifar10
from robustbench.data import load_cifar10c
import numpy as np
import os
from models import *

# 이미지 플롯용
import matplotlib.pyplot as plt
def show(image):
    plt.imshow(image)
    plt.show()

def stat(arr):
    avg = np.sum(arr,axis=0)/arr.shape[0]
    dif = arr - avg
    dif2= dif*dif
    var = np.sum(dif2,axis=0)/arr.shape[0]

    plt.subplot(1,2,1)
    plt.bar(range(0,10),avg)
    plt.xticks(range(0,10),range(0,10))
    plt.title('average')
    
    plt.subplot(1,2,2)
    plt.bar(range(0,10),var)
    plt.xticks(range(0,10),range(0,10))
    plt.title('variance')
    
    plt.show()    

# 모델 세팅
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def modelSet(arch):
    net = arch
    net = net.to(device)
    net.eval()

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    return net

#모델 기존 성능 체크
test_label=1
def test_basic(net):
    x_train,y_train = load_cifar10(n_examples=1000)
    sum=0
    correct =np.empty((0,10))
    err = np.empty((0,10))
    pred_class = np.zeros((10))
    num=0
    for (x,y) in zip(x_train,y_train):
        x,y=x.to(device),y.to(device)
        predict = net(x.unsqueeze(0))
        if(predict.argmax()==y):sum+=1
        if(y==test_label):
            num+=1
            if(predict.argmax()==y): correct=np.append(correct,predict.cpu().detach().numpy(),axis=0)
            elif(predict.argmax()!=y): 
                err=np.append(err,predict.cpu().detach().numpy(),axis=0)
            pred_class[predict.argmax()]+=1
    stat(correct)
    stat(err)
    plt.subplot(1,2,1)
    plt.bar(range(0,10),pred_class)
    plt.xticks(range(0,10),range(0,10))
    plt.title('predicted classes')
    plt.show()
    print('severity: '+str(0)+'(nornal), correct:'+str(sum)+'/1000')
    print('class'+str(test_label)+' count:'+str(num))


#corruption 성능 체크(패턴 확인)
# corroptions = ['gaussian_noise', 'shot_noise', 'impulse_noise',
#                       'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
#                       'snow', 'frost', 'fog', 'brightness', 'contrast',
#                       'elastic_transform', 'pixelate', 'jpeg_compression']
corroptions = ['gaussian_noise']

def test_corroptions(net):
    for corroption in corroptions:
        print('corroption: '+corroption)
        for i in [1]:   #in range(1,6):
            sum=0
            correct =np.empty((0,10))
            err = np.empty((0,10))
            pred_class = np.zeros((10))
            x_train,y_train = load_cifar10c(n_examples=1000,severity=i,corruptions=[corroption])
            num=0
            for (x,y) in zip(x_train,y_train):
                x,y=x.to(device),y.to(device)
                predict = net(x.unsqueeze(0))
                if(predict.argmax()==y): sum+=1
                if(y==test_label):
                    num+=1
                    if(predict.argmax()==y): correct=np.append(correct,predict.cpu().detach().numpy(),axis=0)
                    elif(predict.argmax()!=y): 
                        err=np.append(err,predict.cpu().detach().numpy(),axis=0)
                    pred_class[predict.argmax()]+=1
            stat(correct)
            stat(err)
            plt.subplot(1,2,1)
            plt.bar(range(0,10),pred_class)
            plt.xticks(range(0,10),range(0,10))
            plt.title('predicted classes')
            plt.show()
            print('severity: '+str(i)+', correct:'+str(sum)+'/1000')
            print('class'+str(test_label)+' count:'+str(num))

#Semi-Supervised Error Data Classifier: feature extractor 및 error/correct 클러스터링 클래스
class SSEDC():
    def __init__(self,net) -> None:
        self.sampleBuf= [[[] for j in range(10)] for i in range(2)]    # 2*10 list of empty list
        self.net = net
    
    def calcDist(self, a,b):    # L2 distance
        dist=0
        #normalize
        #an= np.sum(np.abs(a[0])); bn =np.sum(np.abs(b[0]))
        for i in range(len(a[0])):
            #dist+= pow(b[0][i]/bn-a[0][i]/an,2)
            dist+= pow(b[0][i]-a[0][i],2)
        return dist

    def test_err(self, predict, basemodelOutput):
        
        # when no error sample, it's pseudo true(error False)
        if(len(self.sampleBuf[1][predict])==0): return False 
        if(len(self.sampleBuf[0][predict])==0): return False 

        # mean distance for the cluster
        trueDist=0; sampleNum=0
        for trueSample in self.sampleBuf[0][predict]:  #true samples
            trueDist+=self.calcDist(basemodelOutput,trueSample[0])*trueSample[1]    # dist += dist*weight
            sampleNum+=trueSample[1]
        trueDist/=sampleNum
        errDist=0; sampleNum=0
        for errSample in self.sampleBuf[1][predict]:  #error samples
            errDist+=self.calcDist(basemodelOutput,errSample[0])*errSample[1]       # dist += dist*weight
            sampleNum+=errSample[1]
        errDist/=sampleNum
        
        if(trueDist>=errDist):return True
        else: return False
        '''  
               
        # knn(k=5)
        trueDist=[]; sampleNum=0
        for trueSample in self.sampleBuf[0][predict]:  #true samples
            trueDist.append((self.calcDist(basemodelOutput,trueSample[0]),trueSample[1]))
            sampleNum+=trueSample[1]
        
        errDist=[]; sampleNum=0
        for errSample in self.sampleBuf[1][predict]:  #error samples
            errDist.append((self.calcDist(basemodelOutput,errSample[0]),trueSample[1]))
            sampleNum+=errSample[1]
 
        trueDist=sorted(trueDist)
        errDist=sorted(errDist)
        nearest_true=0
        for i in range(5):
            if(nearest_true>len(trueDist)-1 or i-nearest_true>len(errDist)-1): break
            if(trueDist[nearest_true]<=errDist[i-nearest_true]): nearest_true+=1
        if(5-nearest_true<3):return True
        else: return False
        '''

    def test_err_adapt(self, x, y, supervised): #return err length
        basemodelOutput = net(x).cpu().detach().numpy()

        # test
        predict = np.argmax(basemodelOutput)
        err = self.test_err(predict, basemodelOutput)
        
        #semisupervised
        weight=1
        if(supervised):
            err= predict!=y
            weight=3

        # adapt(save basemodelOutput as a sample of error classifier)
        self.sampleBuf[err][predict].append((basemodelOutput,weight))
        if(len(self.sampleBuf[err][predict])>15):self.sampleBuf[err][predict].pop(0)

        return err

def show_score(score_history):
    x_legend=range(score_history.shape[1])
    plt.bar(x_legend,score_history[1],)
    plt.bar(x_legend,score_history[2],bottom=score_history[1])
    plt.bar(x_legend,score_history[3],bottom=np.sum(np.array(score_history)[1:3,:],axis=0))
    plt.bar(x_legend,score_history[4],bottom=np.sum(np.array(score_history)[1:4,:],axis=0))
    ax=plt.subplot()
    ax.set_xticks(x_legend)
    ax.set_xticklabels([str(i) for i in range(score_history.shape[1])])
    plt.title('[modelTruth,ssedcResponse] Result')
    plt.xlabel('iteration')
    plt.ylabel('count')
    plt.legend(['oo','ox','xo','xx'])
    plt.show()

if __name__=="__main__":
    net = modelSet(ResNet50())
    #test_basic(net)
    #test_corroptions(net)
    ssedc = SSEDC(net)
    for corroption in corroptions:
        print('corroption: '+corroption)
        for i in [1]:
            print('severity: '+str(i))
            x_train,y_train = load_cifar10c(n_examples=1000,severity=i,corruptions=[corroption])
            x_train,y_train =x_train.to(device),y_train.to(device)
            sum=0
            num=0
            dataBuffer=[100,199]    #start, end
            semiSuperviseInterval=10
            timing=0
            score_history=[]
            for (x,y) in zip(x_train,y_train):
                dataBuffer[1]+=1
                if(dataBuffer[1]-dataBuffer[0]>100):dataBuffer[0]+=1    #buffer length is 100
                supervised=0
                timing+=1
                if(timing==semiSuperviseInterval):
                    timing=0
                    supervised=1
                ssedc.test_err_adapt(x.unsqueeze(0),y,supervised=supervised)

                #test score 
                score=[0,0,0,0,0] # num, [modelTruth,ssedcResponse]=oo,ox,xo,xx
                for (xi,yi) in zip(x_train[dataBuffer[0]:dataBuffer[1]],y_train[dataBuffer[0]:dataBuffer[1]]):
                    score[0]+=1
                    basemodelOutput = net(xi.unsqueeze(0)).cpu().detach().numpy()
                    predict = np.argmax(basemodelOutput)
                    modelTruth = predict==yi
                    ssedcResponse = not ssedc.test_err(predict,basemodelOutput)
                    score[1] += int(modelTruth and ssedcResponse)
                    score[2] += int(modelTruth and not ssedcResponse)
                    score[3] += int(not modelTruth and ssedcResponse)
                    score[4] += int(not modelTruth and not ssedcResponse)
                print("data_idx: "+str(dataBuffer)+'(len='+str(score[0])+"), ([modelTruth,ssedcResponse]=oo,ox,xo,xx) score: "+str(score[1:]))
                score_history.append(score)
                if(dataBuffer[0]==300):
                    print('here')
                    show_score(np.array(score_history).T)



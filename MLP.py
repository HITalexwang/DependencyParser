# -*- coding: cp936 -*-
import random
import numpy as np
import os
import time

class MLP(object):
    def __init__(self,size,Eb,W1,b1,W2,features=None,labels=None):#size:[50*48,200,|transitions|]
        self.layer_num=len(size)
        self.size=size
        self.b=[np.random.randn(size[1],1) ]
        self.w=[np.random.randn(y,x) for x,y in zip(size[:-1],size[1:])]
        for i in range(len(b1)):
            self.b[0][i]=b1[i]
        self.w[0]=W1
        self.w[1]=W2
        self.Eb=Eb
        if not features==None:
            self.features=features
        if not labels==None:
            self.labels=labels

        self.num_tokens=size[0]/len(Eb[0])
        self.hidden_size=size[1]
        self.embed_size=len(Eb[0])
        self.num_labels=size[2]
        self.reg_parameter=1.0e-8
        self.batch_size=500 #10000
        self.alpha=0.01
        self.ada_eps=1.0e-6
        print size#,self.num_tokens
    """def forward(self,a):
        for b,w in zip(self.b,self.w):
            a=self.sigmoid(np.dot(w,a)+b)
        return a"""

    def bp(self,x,y):
        #forward
        a_list=[x]
        z_list=[]
        for b,w in zip(self.b,self.w):
            z=np.dot(w,a_list[-1])+b
            z_list.append(z)
            a=self.sigmoid(z)
            a_list.append(a)
        #backward
        delta_b=[np.zeros(b.shape) for b in self.b]
        delta_w=[np.zeros(w.shape) for w in self.w]
        delta=(a_list[-1]-y)*self.sigmoid_der(z_list[-1])
        #print delta
        delta_b[-1]=delta
        delta_w[-1]=np.dot(delta,a_list[-2].transpose())
        #print delta_w[-1]
        for i in range(2,self.layer_num):
            z=z_list[-i]
            delta=np.dot(self.w[-i+1].transpose(),delta)*self.sigmoid_der(z)
            delta_b[-i]=delta
            delta_w[-i]=np.dot(delta,a_list[-i-1].transpose())
        return (delta_b,delta_w)
        
        
    def SGD(self,training_data,iter,batch_size,eta,test_data=None):
        if test_data:
            test_num=len(test_data)
        for i in range(iter):
            random.shuffle(training_data)
            batchs=[training_data[j:j+batch_size]
                         for j in range(0,len(training_data),batch_size)]
            for batch in batchs:
                self.update(batch,eta)
            if test_data:
                print "iter",i,"precision:",self.evaluate(test_data),"/",test_num
            else:
                print "iter",i,"finish!"
                
    def update(self,batch,eta):
        delta_b=[np.zeros(b.shape) for b in self.b]
        delta_w=[np.zeros(w.shape) for w in self.w]
        for x,y in batch:
            delta_b_re,delta_w_re=self.bp(x,y)
            #print delta_w_re
            delta_b=[b+re_b for b,re_b in zip(delta_b,delta_b_re)]
            delta_w=[w+re_w for w,re_w in zip(delta_w,delta_w_re)]
        self.w=[w-eta*(del_w/len(batch)) for w,del_w in zip(self.w,delta_w)]
        self.b=[b-eta*(del_b/len(batch)) for b,del_b in zip(self.b,delta_b)]
        
    def sigmoid(self,z):
        return 1.0/(1+np.exp(-z))

    def sigmoid_der(self,z):
        return self.sigmoid(z)*(1-self.sigmoid(z))

    def evaluate(self,test_data):
        results=[(np.argmax(self.forward(x)),y) for (x,y) in test_data]
        return sum(int(x==y) for (x,y) in results)

    def softmax_log(self,a1,label):
        a_max=np.argmax(a1,0)
        #print a1[a_max[0]][0]
        for i in range(len(label)):
            if label[i]==1:
                sum1=np.exp(a1[i][0]-a1[a_max[0]][0])
                #print sum1
        sum2=np.sum([np.exp(ai-a1[a_max[0]][0]) for ai in a1])
        #print sum1
        if sum1==0:
            print "sum1=0,a1:",a1
            raw_in=raw_input("pause")
        return (sum1,sum2)

    def compute_cost_function(self,x,y):
        v_cost=0
        hidden1=np.dot(self.w[0],x)+self.b[0]
        hidden3=np.power(hidden1,3)
        a1=np.dot(self.w[1],hidden3)
        (sum1,sum2)=self.softmax_log(a1,y)
        v_cost+=np.log(sum2)-np.log(sum1)
        v_cost+=self.reg_parameter*np.sum([w*w for w in self.w[0]])/2.0
        v_cost+=self.reg_parameter*np.sum([w*w for w in self.w[1]])/2.0
        v_cost+=self.reg_parameter*np.sum([b*b for b in self.b[0]])/2.0
        return v_cost

    def backprop(self,mini_batch):
        #print self.grad_w[1].shape
        self.loss=0.0
        correct=0
        for i in range(len(mini_batch)):
            #score=[]
            hidden=np.zeros(self.hidden_size)
            hidden3=np.zeros(self.hidden_size)
            (feature,label,original_feat)=mini_batch[i]

            hidden=np.dot(self.w[0],feature)+self.b[0]

            #for 
            #hidden=hidden+self.b[0]
            hidden3=np.power(hidden,3)
            #softmax
            opt_label=-1
            score=np.dot(self.w[1],hidden3)
            #if i==0:
                #print self.w[1][0]
            for j in range(self.num_labels):
                if label[j]>=0:
                    if (opt_label<0 or score[j]>score[opt_label]):
                        opt_label=j
            #loss+=self.softmax_log(score,label)
            (sum1,sum2)=self.softmax_log(score,label)
            self.loss+=np.log(sum2)-np.log(sum1)
            if label[opt_label]==1:
                correct+=1
            #compute gradient
            grad_hidden3=np.zeros(self.hidden_size)
            for i in range(self.num_labels):
                if label[i]>=0:#important???why
                    delta=-(label[i]-score[i][0]/sum2)/len(mini_batch)
                    #print self.grad_w[1].shape
                    #self.grad_w[1][i]+=delta*np.transpose(hidden3)
                    #self.grad_w[1][i]+=delta*hidden3
                    #print hidden3.shape,self.grad_w[1][i].shape
                    for j in range(self.hidden_size):
                        self.grad_w[1][i][j]+=delta*hidden3[j]
                    grad_hidden3+=delta*self.w[1][i]
                #print self.grad_w[1]
            grad_hidden=np.zeros(self.hidden_size)
            #print (np.transpose((hidden*hidden))).shape,grad_hidden3.shape
            #grad_hidden=3*np.dot(np.transpose((hidden*hidden)),grad_hidden3)
            #print grad_hidden.shape
            for j in range(self.hidden_size):
                grad_hidden[j]=grad_hidden3[j]*3*hidden[j]*hidden[j]
                self.grad_b[0][j]+=grad_hidden[j]

            offset=0
            for j in range(self.num_tokens):
                E_index=original_feat[j]
                for k in range(self.hidden_size):
                    for l in range(self.embed_size):
                        self.grad_w[0][k][offset+l] +=grad_hidden[k] * self.Eb[E_index][l];
                        self.grad_Eb[E_index][l] +=grad_hidden[k] * self.w[0][k][offset+l];
                offset+=self.embed_size
        self.loss/=len(mini_batch)
        accuracy=correct/float(len(mini_batch))

        self.add_l2_regularization()
        print "loss:",self.loss,"\naccuracy:",accuracy

    def add_l2_regularization(self):
        self.loss+=self.reg_parameter*np.sum(self.w[0]*self.w[0])/2.0
        self.grad_w[0]+=self.reg_parameter*self.w[0]
        self.loss+=self.reg_parameter*np.sum(self.w[1]*self.w[1])/2.0
        self.grad_w[1]+=self.reg_parameter*self.w[1]
        self.loss+=self.reg_parameter*np.sum(self.b[0]*self.b[0])/2.0
        self.grad_b[0]+=self.reg_parameter*self.b[0]
        self.loss+=self.reg_parameter*np.sum(self.Eb*self.Eb)/2.0
        self.grad_Eb+=self.reg_parameter*self.Eb

    def update(self):
        #print self.grad_w[0]
        self.eg2w[0]+=self.grad_w[0]*self.grad_w[0]
        self.eg2w[1]+=self.grad_w[1]*self.grad_w[1]
        self.eg2b[0]+=self.grad_b[0]*self.grad_b[0]
        self.eg2Eb+=self.grad_Eb*self.grad_Eb

        self.w[0]-=self.alpha*self.grad_w[0]/np.sqrt(self.eg2w[0]+self.ada_eps)
        self.w[1]-=self.alpha*self.grad_w[1]/np.sqrt(self.eg2w[1]+self.ada_eps)
        self.b[0]-=self.alpha*self.grad_b[0]/np.sqrt(self.eg2b[0]+self.ada_eps)
        self.Eb-=self.alpha*self.grad_Eb/np.sqrt(self.eg2Eb+self.ada_eps)

    def train(self,iter):
        start=time.time()
        training_data=self.pre_process()
        print "pre-processing used time:",time.time()-start
        self.grad_w=[np.zeros(w.shape) for w in self.w]
        self.grad_b=[np.zeros(b.shape) for b in self.b]
        self.grad_Eb=np.zeros(self.Eb.shape)

        self.eg2w=[np.zeros(w.shape) for w in self.w]
        self.eg2b=[np.zeros(b.shape) for b in self.b]
        self.eg2Eb=np.zeros(self.Eb.shape)
        print len(training_data)
        for i in range(iter):
            print "iter ",i
            random.shuffle(training_data)
            batchs=[training_data[j:j+self.batch_size]
                         for j in range(0,len(training_data),self.batch_size)]
            for batch in batchs:
                start1=time.time()
                batch=self.pre_process_batch(batch)
                self.backprop(batch)
                print "back propogation used time:",time.time()-start1
                self.update()
                self.grad_w[0]*=0
                self.grad_w[1]*=0
                self.grad_b[0]*=0
                self.grad_Eb*=0
            training_data=self.pre_process()

    def pre_process(self):
        training_data=[]
        for i in range(len(self.features)):
            training_data.append((self.labels[i],self.features[i]))
        return training_data

    def pre_process_batch(self,batch):
        training_data=[]
        for i in range(len(batch)):
            (label,feature)=batch[i]
            offset=0
            x=np.zeros([self.size[0],1])
            for j in range(len(feature)):
                for k in range(offset,offset+self.embed_size):
                    x[k]=self.Eb[feature[j]][k-offset]
                offset+=self.embed_size
            training_data.append((x,label,feature))
        #print "batch len:",len(training_data)
        return training_data

    def get_w1(self):
        return self.w[0]

    def get_b1(self):
        return self.b[0]

    def get_w2(self):
        return self.w[1]

    def get_Eb(self):
        return self.Eb

    def compute_scores(self,features):
        #scores=np.zeros(self.size[2])
        hidden=np.zeros([self.hidden_size,1])
        offset=0
        for i in range(len(features)):
            E_index=features[i]
            for j in range(self.hidden_size):
                for k in range(self.embed_size):
                    hidden[j]+=self.Eb[E_index][k]*self.w[0][j][offset+k]
            offset+=self.embed_size

        hidden+=self.b[0]
        hidden=hidden*hidden*hidden
        scores=np.dot(self.w[1],hidden)
        return scores

#if __name__=="__main__":
    #classifier=MLP([784,30,10])
    #classifier.SGD(training_data,30,10,3.0,test_data=test_data)
    #classifier=MLP([3,5,4])

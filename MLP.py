# -*- coding: cp936 -*-
import random
import numpy as np
import os
import time
import threading
import multiprocessing

class Cost:
    def __init__(self,grad_Eb,grad_W1,grad_b1,grad_W2,loss,correct):
        self.grad_Eb=grad_Eb
        self.grad_W1=grad_W1
        self.grad_W2=grad_W2
        self.grad_b1=grad_b1
        self.loss=loss
        self.correct=correct

class MLP(object):
    def __init__(self,size,Eb,W1,b1,W2,pre_computed_ids=None,features=None,labels=None):#size:[50*48,200,|transitions|]
        self.layer_num=len(size)
        self.size=size
        self.b=[np.random.randn(size[1],1) ]
        self.w=[np.random.randn(y,x) for x,y in zip(size[:-1],size[1:])]
        for i in range(len(b1)):
            self.b[0][i]=b1[i]
        self.w[0]=W1
        self.w[1]=W2
        self.Eb=Eb

        self.num_tokens=size[0]/len(Eb[0])
        self.hidden_size=size[1]
        self.embed_size=len(Eb[0])
        self.num_labels=size[2]
        self.reg_parameter=1.0e-8
        self.batch_size=10000 #10000
        self.alpha=0.01
        self.ada_eps=1.0e-6
        self.training_threads=10
        self.trunk_size=self.batch_size/self.training_threads

        if not pre_computed_ids==None:
            self.pre_computed_ids=pre_computed_ids
            self.pre_map={}
            for i in range(len(pre_computed_ids)):
                self.pre_map[pre_computed_ids[i]]=i
            self.grad_saved=np.zeros([len(self.pre_map),self.hidden_size])
        if not features==None:
            self.features=features
        if not labels==None:
            self.labels=labels

        self.pos_emb_size=10
        self.label_emb_size=10
        
        print size#,self.num_tokens
    """def forward(self,a):
        for b,w in zip(self.b,self.w):
            a=self.sigmoid(np.dot(w,a)+b)
        return a"""

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

    def backprop(self,mini_batch,costs):
        #print self.grad_w[1].shape
        loss=0.0
        correct=0
        grad_w=[np.zeros(w.shape) for w in self.w]
        grad_b=[np.zeros(b.shape) for b in self.b]
        grad_Eb=np.zeros(self.Eb.shape)

        hidden=np.zeros([self.hidden_size,1])
        hidden3=np.zeros([self.hidden_size,1])
        grad_hidden=np.zeros([self.hidden_size,1])
        grad_hidden3=np.zeros([self.hidden_size,1])
        mini_batch_size=len(mini_batch)
        for i in range(mini_batch_size):
            #score=[]
            hidden*=0
            hidden3*=0
            (label,feature)=mini_batch[i]

            #hidden=np.dot(self.w[0],feature)+self.b[0]
            offset=0

            time1=time.time()

            for j in range(self.num_tokens):
                tok=feature[j]
                E_index=tok
                index=tok*self.num_tokens+j
                if index in self.pre_map:
                    id=self.pre_map[index]
                    """
                    for k in range(self.hidden_size):
                        hidden[k][0]+=self.saved[id][k]
                    """
                    hidden[:,0]+=np.transpose(self.saved[id][:])
                else:
                    """
                    for k in range(self.hidden_size):
                        for l in range(self.embed_size):
                            hidden[k][0]+=self.w[0][k][offset+l]*self.Eb[E_index][l]
                    """
                    hidden[:,0]+=np.dot(self.w[0][:,offset:offset+self.embed_size],np.transpose(self.Eb[E_index,:]))
                offset+=self.embed_size

            time2=time.time()
           # print "claculate Eb*w1 used",time2-time1

            hidden=hidden+self.b[0]

            time3=time.time()
            #print "calculate +b1 used",time3-time2

            hidden3=np.power(hidden,3)

            time4=time.time()
            #print "calculate power 3 used",time4-time3

            #softmax
            opt_label=-1
            #print hidden3.shape
            score=np.dot(self.w[1],hidden3)
            #print score.shape
            time5=time.time()
            #print "calculate score used",time5-time4
            #if i==0:
                #print self.w[1][0]
            for j in range(self.num_labels):
                if label[j]>=0:
                    if (opt_label<0 or score[j][0]>score[opt_label][0]):
                        opt_label=j

            #(sum1,sum2)=self.softmax_log(score,label)
            max_score=score[opt_label][0]
            sum1=0
            sum2=0
            for j in range(self.num_labels):
                if label[j]>=0:
                    score[j][0]=np.exp(score[j][0]-max_score)
                    if label[j]==1:
                        sum1+=score[j][0]
                    sum2+=score[j][0]
            if sum1==0:
                print "opt_label=",opt_label
                print "max_score=",max_score
                print "score:",score
                raw_input("pause")
            time6=time.time()
            #print "calculate softmax_log used",time6-time5

            loss+=(np.log(sum2)-np.log(sum1))
            if label[opt_label]==1:
                correct+=1
            #compute gradient
            grad_hidden3*=0
            for i in range(self.num_labels):
                if label[i]>=0:#important???why
                    delta=-(label[i]-score[i][0]/sum2)/mini_batch_size
                    #print self.grad_w[1].shape
                    #self.grad_w[1][i]+=delta*np.transpose(hidden3)
                    #self.grad_w[1][i]+=delta*hidden3
                    """
                    for j in range(self.hidden_size):
                        grad_w[1][i][j]+=delta*hidden3[j][0]
                        grad_hidden3[j][0]+=delta*self.w[1][i][j]
                    """
                    grad_w[1][i,:]+=delta*hidden3[:,0]
                    grad_hidden3[:,0]+=delta*self.w[1][i,:]


            time7=time.time()
            #print "calculate delta used",time7-time6

            grad_hidden*=0
            #print (np.transpose((hidden*hidden))).shape,grad_hidden3.shape
            #grad_hidden=3*np.dot(np.transpose((hidden*hidden)),grad_hidden3)
            #print grad_hidden.shape
            """
            for j in range(self.hidden_size):
                grad_hidden[j][0]=grad_hidden3[j][0]*3*hidden[j][0]*hidden[j][0]
                grad_b[0][j][0]+=grad_hidden[j][0]
            """
            grad_hidden=grad_hidden3*3*hidden*hidden
            grad_b[0]+=grad_hidden
            time8=time.time()
            #print "calculate grad hidden used",time8-time7

            offset=0
            for j in range(self.num_tokens):
                tok=feature[j]
                E_index=tok
                index=tok*self.num_tokens+j

                if index in self.pre_map:
                    id=self.pre_map[index]
                    """
                    for k in range(self.hidden_size):
                        self.grad_saved[id][k]+=grad_hidden[k][0]
                    """
                    self.grad_saved[id,:]+=np.transpose(grad_hidden[:,0])
                else:
                    """
                    for k in range(self.hidden_size):
                        for l in range(self.embed_size):
                            grad_w[0][k][offset+l] +=grad_hidden[k][0] * self.Eb[E_index][l];
                            grad_Eb[E_index][l] +=grad_hidden[k][0] * self.w[0][k][offset+l];
                    """
                    grad_w[0][:,offset,offset+self.embed_size]+=np.outer(grad_hidden[:,0],self.Eb[E_index,:])
                    grad_Eb[E_index,:]+=np.dot(np.transpose(grad_hidden[:,0]),self.w[0][:,offset,offset+self.embed_size])
                offset+=self.embed_size

            time9=time.time()
            #print "calculate grad eb used",time9-time8

        loss/=len(mini_batch)
        #accuracy=correct/float(mini_batch_size)
        cost=Cost(grad_Eb,grad_w[0],grad_b[0],grad_w[1],loss,correct)
        #self.costs.append(cost)
        #costs.put((grad_Eb,grad_w[1],grad_b[0],loss,correct,grad_w[0]))
        costs.put(cost)

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

        check=False
        if check:
            self.check_gradient(training_data)
        else:
            for i in range(iter):
                print "iter ",i
                random.shuffle(training_data)
                batchs=[training_data[j:j+self.batch_size]
                                for j in range(0,len(training_data),self.batch_size)]
                for batch in batchs:
                    self.compute_cost_function(batch)
                    self.update()
                    self.grad_w[0]*=0
                    self.grad_w[1]*=0
                    self.grad_b[0]*=0
                    self.grad_Eb*=0
            #training_data=self.pre_process()
        
        """
                start1=time.time()
                self.pre_compute(self.get_pre_computed_ids(batch))
                time2=time.time()
                print "pre_computing time:",time2-start1
                #batch=self.pre_process_batch(batch)
                self.backprop(batch)
                print "back propogation used time in totol:",time.time()-time2
                """

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

    def get_pre_computed_ids(self,batch):
        feature_ids=[]
        for i in range(len(batch)):
            (label,feature)=batch[i]
            for j in range(len(feature)):
                tok=feature[j]
                index=tok*self.num_tokens+j
                if ((index in self.pre_map) and (index not in feature_ids)):
                    feature_ids.append(index)
        return feature_ids

    def pre_compute(self,candidates):
        print "pre_map size=",len(self.pre_map)
        print "candidates size=",len(candidates)

        self.saved=np.zeros([len(self.pre_map),self.hidden_size])
        for i in range(len(candidates)):
            map_x=self.pre_map[candidates[i]]
            tok=candidates[i]/self.num_tokens
            pos=candidates[i]%self.num_tokens
            offset=pos*self.embed_size

            E_index=tok
            """
            for j in range(self.hidden_size):
                for k in range(self.embed_size):
                    self.saved[map_x][j]+=self.Eb[E_index][k]*self.w[0][j][offset+k]
            """
            self.saved[map_x,:]=np.dot(self.w[0][:,offset:offset+self.embed_size],np.transpose(self.Eb[E_index,:]))
        print "pre_computed ",len(candidates)

    def back_prop_saved(self,features_seen):
        for i in range(len(features_seen)):
            map_x=self.pre_map[features_seen[i]]
            tok=features_seen[i]/self.num_tokens
            pos=features_seen[i]%self.num_tokens
            offset=pos*self.embed_size

            E_index=tok
            self.grad_w[0][:,offset:offset+self.embed_size]=np.outer(self.grad_saved[map_x,:],self.Eb[E_index,:])
            self.grad_Eb[E_index,:]+=np.dot(self.grad_saved[map_x,:],self.w[0][:,offset:offset+self.embed_size])
            """
            for j in range(self.hidden_size):
                delta=self.grad_saved[map_x][j]
                for k in range(self.embed_size):
                    self.grad_w[0][j][offset+k]+=delta*self.Eb[E_index][k]
                    self.grad_Eb[E_index][k]+=delta*self.w[0][j][offset+k]
            """


    def check_gradient(self,batch):
        print "---checking gradient---"
        self.compute_cost_function(batch)
        self.compute_numerical_gradient(batch)
        diff_grad_w1=np.sum(np.power(self.num_grad_w[0]-self.grad_w[0],2))/np.sum(np.power(self.num_grad_w[0]+self.grad_w[0],2))
        diff_grad_b1=np.sum(np.power(self.num_grad_b[0]-self.grad_b[0],2))/np.sum(np.power(self.num_grad_b[0]+self.grad_b[0],2))
        diff_grad_w2=np.sum(np.power(self.num_grad_w[1]-self.grad_w[1],2))/np.sum(np.power(self.num_grad_w[1]+self.grad_w[1],2))
        diff_grad_Eb=np.sum(np.power(self.num_grad_Eb-self.grad_Eb,2))/np.sum(np.power(self.num_grad_Eb+self.grad_Eb,2))
        print "diff w1:",diff_grad_w1
        print "diff b1:",diff_grad_b1
        print "diff w2:",diff_grad_w2
        print "diff Eb:",diff_grad_Eb

    def compute_cost_function(self,batch):
        self.costs=[]
        self.loss=0
        self.correct=0
        time1=time.time()
        pre_computed_ids=self.get_pre_computed_ids(batch)
        self.pre_compute(pre_computed_ids)
        time2=time.time()
        print "pre_compute:",time2-time1
        self.grad_saved*=0
        
        trunks=[batch[j:j+self.trunk_size]
                         for j in range(0,len(batch),self.trunk_size)]
        time2_5=time.time()
        print "---trunks length:",len(trunks),"---"
        costs=multiprocessing.Queue(self.training_threads)
        process_pool=[]
        #lock = multiprocessing.Lock()
        mgr=multiprocessing.Manager()
        costs=mgr.Queue()
        try:
            for trunk in trunks:
                pr=multiprocessing.Process(target=self.backprop,args=(trunk,costs))
                pr.start()
                process_pool.append(pr)
        except:
            print "Error: unable to start thread"
        for process in process_pool:
            process.join()

        print "#thread 10 time:",time.time()-time2_5

        for i in range(len(trunks)):
            self.costs.append(costs.get())
        print "---costs length:",len(self.costs),"---"
        #self.backprop(batch)
        for cost in self.costs:
            self.merge_cost(cost)
        self.loss/=len(self.costs)

        time3=time.time()
        print "back prop:",time3-time2
        self.add_l2_regularization()
        print "loss:",self.loss,"\naccuracy:",float(self.correct)/len(batch)
        self.back_prop_saved(pre_computed_ids)

    def merge_cost(self,cost):
        self.grad_w[0]+=cost.grad_W1
        self.grad_w[1]+=cost.grad_W2
        self.grad_b[0]+=cost.grad_b1
        self.grad_Eb+=cost.grad_Eb
        self.loss+=cost.loss
        self.correct+=cost.correct


    def compute_numerical_gradient(self,batch):
        self.num_grad_w=[np.zeros(w.shape) for w in self.w]
        self.num_grad_b=[np.zeros(b.shape) for b in self.b]
        self.num_grad_Eb=np.zeros(self.Eb.shape)
        epsilon=1e-6
        print "checking w1"
        for i in range(len(self.w[0])):
            for j in range(len(self.w[0][0])):
                self.w[0][i][j]+=epsilon
                p_eps_cost=self.compute_cost(batch)
                self.w[0][i][j]-=2*epsilon
                n_eps_cost=self.compute_cost(batch)
                self.num_grad_w[0][i][j]= (p_eps_cost - n_eps_cost) / (2 * epsilon)
                self.w[0][i][j]+=epsilon
        print "checking b1"
        for i in range(len(self.b[0])):
            self.b[0][i]+=epsilon
            p_eps_cost=self.compute_cost(batch)
            self.b[0][i]-=2*epsilon
            n_eps_cost=self.compute_cost(batch)
            self.num_grad_b[0][i]= (p_eps_cost - n_eps_cost) / (2 * epsilon)
            self.b[0][i]+=epsilon
        print "checking w2"
        for i in range(len(self.w[1])):
            for j in range(len(self.w[1][0])):
                self.w[1][i][j]+=epsilon
                p_eps_cost=self.compute_cost(batch)
                self.w[1][i][j]-=2*epsilon
                n_eps_cost=self.compute_cost(batch)
                self.num_grad_w[1][i][j]= (p_eps_cost - n_eps_cost) / (2 * epsilon)
                self.w[1][i][j]+=epsilon
        print "checking Eb"
        for i in range(len(self.Eb)):
            for j in range(len(self.Eb[0])):
                self.Eb[i][j]+=epsilon
                p_eps_cost=self.compute_cost(batch)
                self.Eb[i][j]-=2*epsilon
                n_eps_cost=self.compute_cost(batch)
                self.num_grad_Eb[i][j]= (p_eps_cost - n_eps_cost) / (2 * epsilon)
                self.Eb[i][j]+=epsilon

    def compute_cost(self,batch):
        v_cost=0
        for i in range(len(batch)):
            hidden=np.zeros([self.hidden_size,1])
            (label,feature)=batch[i]
            offset=0
            for j in range(self.num_tokens):
                tok=feature[j]
                E_index=tok
                for k in range(self.hidden_size):
                    for l in range(self.embed_size):
                        hidden[k]+=self.w[0][k][offset+l]*self.Eb[E_index][l]
                offset+=self.embed_size
            hidden=hidden+self.b[0]
            #hidden1=np.dot(self.w[0],x)+self.b[0]
            hidden3=np.power(hidden,3)
            a1=np.dot(self.w[1],hidden3)
            (sum1,sum2)=self.softmax_log(a1,label)
            v_cost+=np.log(sum2)-np.log(sum1)
        v_cost/=len(batch)
        v_cost+=self.reg_parameter*np.sum(self.w[0]*self.w[0])/2.0
        v_cost+=self.reg_parameter*np.sum(self.w[1]*self.w[1])/2.0
        v_cost+=self.reg_parameter*np.sum(self.b[0]*self.b[0])/2.0
        v_cost+=self.reg_parameter*np.sum(self.Eb*self.Eb)/2.0
        return v_cost

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

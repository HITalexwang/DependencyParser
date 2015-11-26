#!/usr/bin/python
#coding=utf-8
import numpy as np
import copy
import MLP
import os

class ArcStandard:
	def __init__(self,ldict,language,is_labeled):
		self.labeled=is_labeled
		self.labels=ldict
		self.lang=language
		self.root_label=self.labels[-1]
		self.make_transitions()
		print "---ArcStandard---"
		print "#transitions:",len(self.transitions)
		print "#labels:",len(self.labels)
		print "#ROOT label:",self.root_label

	def make_transitions(self):
		self.transitions=[]
		for i in range(len(self.labels)):
			self.transitions.append('L('+self.labels[i]+')')
		for i in range(len(self.labels)):
			self.transitions.append('R('+self.labels[i]+')')
		self.transitions.append('S')
		"""
		print "transition types:"
		for i in range(len(self.transitions)):
			print selfself.transitions[i]
		"""
	def can_apply(self,c,t):#configuration c
		if (t.startswith('R') or t.startswith('L')):
			label=t[2:-1]
			if t.startswith("L"):
				h=c.get_stack(0)
			else:
				h=c.get_stack(1)
			if h<0:
				return False
			if self.labeled:
				if (h==0 and (not label==self.root_label)):
					return False
				if (h>0 and label==self.root_label):
					return False
		n_stack=c.get_stack_size()
		n_buffer=c.get_buffer_size()
		if t.startswith("L"):
			#print "can apply L",n_stack
			return n_stack>2
		elif t.startswith("R"):
			#print "can apply R",n_stack,n_buffer
			return (n_stack>2 or (n_stack==2 and n_buffer==0))
		else: #shift
			#print "can apply S",n_buffer
			return n_buffer>0

	def apply(self,c,t):#configuration c
		w1=c.get_stack(1)
		w2=c.get_stack(0)
		#left
		if t.startswith("L"):
			c.add_arc(w2,w1,t[2:-1])
			c.remove_second_top_stack()
			#c.lvalency[w2] += 1;
		#right
		elif t.startswith("R"):
			c.add_arc(w1,w2,t[2:-1])
			c.remove_top_stack()
		else:
			c.shift()

	def get_oracle(self,c,tree):#configuration c
		w1=c.get_stack(1)
		w2=c.get_stack(0)
		if (w1>0 and tree.get_head(w1)==w2):
			return "L("+tree.get_label(w1)+")"
		elif (w1>=0 and tree.get_head(w2)==w1 and (not c.has_other_child(w2,tree))):
			return "R("+tree.get_label(w2)+")"
		else:
			return "S"

	def init_configuration(self,sent):
		c=Configuration(sent)
		return c

	def is_terminal(self,c):#configuration c
		return (c.get_buffer_size()==0 and c.get_stack_size()==1)

class DependencySent:
	def __init__(self):
		self.n=0
		self.words=[]
		self.poss=[]
		self.clusters=[]

	def add(self,word,pos,cluster):
		self.n+=1
		self.words.append(word)
		self.poss.append(pos)
		self.clusters.append(cluster)

	def print_info(self):
		print "n=",self.n
		print self.words,',',self.poss,',',self.clusters

class DependencyTree:
	def __init__(self):
		self.n=0
		self.heads=[-1]
		self.labels=['-NULL-']#used to be '-UNKNOWN-'???

	def add(self,h,l):
		self.n+=1
		self.heads.append(h)
		self.labels.append(l)

	def set(self,k,h,l):
		self.heads[k]=h
		self.labels[k]=l

	def get_head(self,k):
		if (k<=0 or k>self.n):
			return -1
		else:
			return self.heads[k]

	def get_label(self,k):
		if(k<=0 or k>self.n):
			return '-NULL-'
		else:
			return self.labels[k]

	def get_root(self):
		for k in range(1,self.n+1):
			if self.get_head(k)==0:
				return k
		return 0 #non tree

	def is_single_root(self):
		roots=0
		for k in range(1,self.n+1):
			if self.get_head(k)==0:
				roots+=1
		return (roots==1)

	def is_tree(self):
		h=[-1] 
		for k in range(1,self.n+1):
			if (self.get_head(k)<0 or self.get_head(k)>self.n):
				return False
			h.append(-1)
		for k in range(1,self.n+1):
			i=k
			while (i>0):
				if (h[i]>=0 and h[i]<k):
					break
				if(h[i]==k):
					return False
				h[i]=k #h[i]=k  means the original child is k,so if h[i]==k means it returns to the origin,which makes a circle
				i=self.get_head(i)
		return True

	def is_projective(self):
		if (not self.is_tree()):
			return False
		self.counter=-1
		return self.visit_tree(0)

	def visit_tree(self,w): #??? dont understand
		for k in range(1,w):
			if (self.get_head(k)==w and self.visit_tree(k)==False):
				return False
		self.counter+=1
		if (w!=self.counter):
			return False
		for k in range(w+1,self.n+1):
			if (self.get_head(k)==w and self.visit_tree(k)==False):
				return False
		return True

	def print_tree(self):
		for i in range(0,self.n+1):
			print i,self.get_head(i),',',self.get_label(i)

class Parser:
	def __init__(self):
		self.word_ids={}
		self.label_ids={}
		self.pos_ids={}
		self.pre_computed_ids={}
		self.delexicalized=False
		self.use_postag=True
		self.labeled=True
		self.num_tokens=48
		self.embedding_size=50
		self.num_pre_computed=100000
		self.hidden_size=50
		self.pos_emb_size=10
		self.label_emb_size=10

	def load_file(self,file,sents,trees,labeled):
		sent=DependencySent()
		tree=DependencyTree()
		data=open(file)
		for line in data:
			sep_line=line.strip().split()
			#print sep_line
			if len(sep_line)<10:
				sents.append(sent)
				trees.append(tree)
				sent=DependencySent()
				tree=DependencyTree()
			else:
				word=sep_line[1]
				pos=sep_line[3]
				cluster=sep_line[5]
				deprel=sep_line[7]
				head=int(sep_line[6])
				sent.add(word,pos,cluster)
				if labeled:
					tree.add(head,deprel)
				else:
					tree.add(head,'-UNKNOWN-')
		data.close()

	def print_tree_states(self,trees):
		print "there are ",len(trees),"trees"
		non_trees=0
		non_proj=0
		for i in range(len(trees)):
			if not trees[i].is_tree():
				non_trees+=1
			elif not trees[i].is_projective():
				non_proj+=1
		print "---tree states---"
		print non_trees,"trees are illegal"
		print non_proj,"trees are not projective"

	def read_embed_file(self,embed_file):
		embed_data=open(embed_file)
		embed_ids={}
		embeddings=[]
		word_cnt=0
		for line in embed_data:
			if len(line)==0:break
			sep_line=line.strip().split()
			embed_ids[sep_line[0]]=word_cnt
			embedding=[float(emb) for emb in sep_line[1:]]
			embeddings.append(embedding)
			word_cnt+=1
		self.embed_ids=embed_ids
		self.embeddings=embeddings
		return (embed_ids,embeddings)

	def gen_dictionaries(self,sents,trees,labeled):
		known_words=[]
		known_poss=[]
		known_labels=[]
		for i in range(len(sents)):
			for w in sents[i].words:
				if w not in known_words:
					known_words.append(w)
			for p in sents[i].poss:
				if p not in known_poss:
					known_poss.append(p)
		if labeled:
			for j in range(len(trees)):
				#trees[j].print_tree()
				for k in range(1,trees[j].n+1):
					if (trees[j].get_head(k)==0):
						root_label=trees[j].get_label(k)
					elif trees[j].get_label(k) not in known_labels:
						#print trees[j].get_label(k)
						known_labels.append(trees[j].get_label(k))
		known_words.append('-UNKNOWN-')
		known_words.append('-NULL-')
		known_words.append('-ROOT-')
		known_poss.append('-UNKNOWN-')
		known_poss.append('-NULL-')
		known_poss.append('-ROOT-')
		if labeled:
			known_labels.append('-NULL-')
			if root_label not in known_labels:
				known_labels.append(root_label)
		else:
			known_labels.append('-UNKNOWN-')
		self.known_labels=known_labels
		self.known_words=known_words
		self.known_poss=known_poss
		self.generate_ids()
		print "---Dictionaries---"
		print "words:",len(known_words)
		print "poss:",len(known_poss)
		if labeled:
			print "labels:",len(known_labels)
		return (known_words,known_poss,known_labels)

	def generate_ids(self):
		index=0
		#print "labels:",self.known_labels
		for i in range(len(self.known_words)):
			self.word_ids[self.known_words[i]]=index
			index+=1
		if self.delexicalized:
			index=0
		for i in range(len(self.known_poss)):
			if not self.use_postag:
				self.pos_ids[self.known_poss[i]]=0
			else:
				self.pos_ids[self.known_poss[i]]=index
				index+=1
		for i in range(len(self.known_labels)):
			self.label_ids[self.known_labels[i]]=index
			index+=1

	def setup_classifier_for_trainning(self,sents,trees,labeled):
		embedding_size=self.embedding_size
		num_basic_tokens=self.num_tokens #num_word_tokens  = 18; num_pos_tokens= 18; num_label_tokens = 12;
		hidden_size=self.hidden_size
		init_range=0.010
		Eb_entries=0
		Eb_entries=len(self.known_poss)+len(self.known_words)
		if labeled:
			Eb_entries+=len(self.known_labels)
		#Eb=np.zeros([Eb_entries,embedding_size],float) 
		Eb=np.random.rand(Eb_entries,embedding_size)#init Embeddings randomly for words,poss and labels
		Eb=(Eb*2-1)*init_range
		W1_ncol=embedding_size*num_basic_tokens
		W1_init_range=np.sqrt(6.0/(W1_ncol+hidden_size))
		#W1=np.zeros([hidden_size,W1_ncol],float)
		W1=np.random.rand(hidden_size,W1_ncol)
		W1=(W1*2-1)*W1_init_range
		#b1=np.zeros(hidden_size,float)
		b1=np.random.rand(hidden_size)
		b1=(b1*2-1)*W1_init_range
		if labeled:
			n_actions=len(self.known_labels)*2+1
		else:
			n_actions=3
		W2_init_range=np.sqrt(6.0/(n_actions+hidden_size))
		#W2=np.zeros([n_actions,hidden_size],float)
		W2=np.random.rand(n_actions,hidden_size)
		W2=(W2*2-1)*W2_init_range

		#match embeding dictionary
		in_embed=0
		Eb_nrows=Eb_entries
		Eb_ncols=embedding_size
		for i in range(Eb_nrows):
			index=-1
			if (i<len(self.known_words)):
				#print self.known_words[i]
				word=self.known_words[i]
				if word in self.embed_ids:
					index=self.embed_ids[word]
				elif word.lower() in self.embed_ids:
					index=self.embed_ids[word.lower()]
			if index>=0:
				in_embed+=1
				Eb[i]=self.embeddings[index]
		print "---Setup Classifier---"
		print "found embeddings:",in_embed,"/",len(self.known_words)
		dataset=self.gen_train_samples(sents,trees)
		print "create classifier"
		#classifier=NNClassifier(dataset,Eb,W1,b1,W2,self.pre_computed_ids)
		(features,labels)=self.preprocess_dataset(dataset)
		self.classifier=MLP.MLP([self.embedding_size*self.num_tokens,self.hidden_size,n_actions],Eb,W1,b1,W2,self.pre_computed_ids,features,labels)

	def preprocess_dataset(self,dataset):
		features=[]
		labels=[]
		for sample in dataset.samples:
			features.append(sample.get_feature())
			labels.append(sample.get_label())
		return (features,labels)

	def gen_train_samples(self,sents,trees):
		num_tokens=48
		num_trans=len(self.system.transitions)
		ds_train=Dataset(num_tokens,num_trans)
		print "generating training examples"
		tokpos_count={}
		for i in range(len(sents)):
			if i%1000==0:
				print "iter",i
		#only use projective tree
			if trees[i].is_projective():
				c=Configuration(sents[i])
				while(not self.system.is_terminal(c)):
					oracle=self.system.get_oracle(c,trees[i])
					features=self.get_features(c)
					#print features
					label=[]
					for k in range(num_trans):
						label.append(-1)
					for j in range(num_trans):
						action=self.system.transitions[j]
						if action==oracle:
							label[j]=1
						elif self.system.can_apply(c,action):
							label[j]=0
					ds_train.add_sample(features,label)
					for j in range(len(features)):
						feature_id=features[j]*len(features)+j
						if feature_id not in tokpos_count:
							tokpos_count[feature_id]=1
						else:
							tokpos_count[feature_id]+=1
					self.system.apply(c,oracle)
		print "#train examples num:",ds_train.n
		temp=copy.deepcopy(tokpos_count)
		print "sort tokpos_count"
		temp=sorted(temp.iteritems(),key=lambda d:d[1],reverse=True)
		self.pre_computed_ids=[]
		print "fill pre_computed_ids"
		real_size=min(len(temp),self.num_pre_computed)
		cnt=0
		for t in temp:
			self.pre_computed_ids.append(t[0])
			cnt+=1
			if cnt>=real_size:
				break
		#print self.pre_computed_ids
		return ds_train


	def get_features(self,c):#configuration c
		f_word=[]
		f_pos=[]
		f_label=[]
		for i in range(2,-1,-1):#stack 2,1,0,tokens=3
			index=c.get_stack(i)
			f_word.append(self.get_word_id(c.get_word(index)))
			f_pos.append(self.get_pos_id(c.get_pos(index)))
		for i in range(3):#buffer 0,1,2,tokens=3
			index=c.get_buffer(i)
			f_word.append(self.get_word_id(c.get_word(index)))
			f_pos.append(self.get_pos_id(c.get_pos(index)))
		for i in range(2):#stack 0,1,tokens=2*6=12
			k=c.get_stack(i)
			index=c.get_left_child(k,1)
			f_word.append(self.get_word_id(c.get_word(index)))
			f_pos.append(self.get_pos_id(c.get_pos(index)))
			f_label.append(self.get_label_id(c.get_label(index)))

			index=c.get_right_child(k,1)
			f_word.append(self.get_word_id(c.get_word(index)))
			f_pos.append(self.get_pos_id(c.get_pos(index)))
			f_label.append(self.get_label_id(c.get_label(index)))

			index=c.get_left_child(k,2)
			f_word.append(self.get_word_id(c.get_word(index)))
			f_pos.append(self.get_pos_id(c.get_pos(index)))
			f_label.append(self.get_label_id(c.get_label(index)))

			index=c.get_right_child(k,2)
			f_word.append(self.get_word_id(c.get_word(index)))
			f_pos.append(self.get_pos_id(c.get_pos(index)))
			f_label.append(self.get_label_id(c.get_label(index)))

			index=c.get_left_child(c.get_left_child(k,1),1)
			f_word.append(self.get_word_id(c.get_word(index)))
			f_pos.append(self.get_pos_id(c.get_pos(index)))
			f_label.append(self.get_label_id(c.get_label(index)))

			index=c.get_right_child(c.get_right_child(k,1),1)
			f_word.append(self.get_word_id(c.get_word(index)))
			f_pos.append(self.get_pos_id(c.get_pos(index)))
			f_label.append(self.get_label_id(c.get_label(index)))

		features=[]
		if not self.delexicalized:
			features+=f_word
		if self.use_postag:
			features+=f_pos
		if self.labeled:
			features+=f_label 
		if not len(features)==self.num_tokens:
			print "----number of tokens wrong!----"
			print "feat:",len(features)
		return features

	def get_word_id(self,s):#string s
		if s not in self.word_ids:
			if s.lower() in self.word_ids:
				return self.word_ids[s.lower()]
			else:
				if "-UNKNOWN-" not in self.word_ids:
					return -1
				else:
					return self.word_ids["-UNKNOWN-"]
		else:
			return self.word_ids[s]

	def get_pos_id(self,s):#string s
		if s not in self.pos_ids:
			return self.pos_ids["-UNKNOWN-"]
		else:
			return self.pos_ids[s]

	def get_label_id(self,s):#string s
		return self.label_ids[s]

	def save_model(self,filename):
		w1=self.classifier.get_w1()
		w2=self.classifier.get_w2()
		b1=self.classifier.get_b1()
		Eb=self.classifier.get_Eb()
		if os.path.exists(filename):
			os.remove(filename)
		model=open(filename,'a')
		model.write("dict="+str(len(self.known_words))+"\n")
		model.write("pos="+str(len(self.known_poss))+"\n")
		model.write("label="+str(len(self.known_labels))+"\n")
		model.write("embedding_size="+str(self.embedding_size)+"\n")
		model.write("hidden_size="+str(self.hidden_size)+"\n")
		model.write("basic_tokens="+str(self.num_tokens)+"\n")

		#write word/pos/label
		for i in range(len(self.known_words)):
			model.write(str(self.known_words[i])+" ")
		model.write("\n")
		for i in range(len(self.known_poss)):
			model.write(str(self.known_poss[i])+" ")
		model.write("\n")
		for i in range(len(self.known_labels)):
			model.write(str(self.known_labels[i])+" ")
		model.write("\n")
		#write word/pos/label embeddings
		for i in range(len(Eb)):
			for j in range(len(Eb[0])):
				model.write(str(Eb[i][j])+" ")
			model.write("\n")
		model.write("\n")
		#write w1
		for i in range(len(w1)):
			for j in range(len(w1[0])):
				model.write(str(w1[i][j])+" ")
			model.write("\n")
		model.write("\n")
		#write b1
		for i in range(len(b1)):
			model.write(str(b1[i][0])+" ")
		model.write("\n\n")
		#write w2
		for i in range(len(w2)):
			for j in range(len(w2[0])):
				model.write(str(w2[i][j])+" ")
			model.write("\n")
		model.close()

	def load_model(self,model_file):
		print "loading model from file:",model_file
		data=open(model_file)
		n_dict=int(data.readline().split("=")[1])
		n_pos=int(data.readline().split("=")[1])
		n_label=int(data.readline().split("=")[1])
		self.embedding_size=int(data.readline().split("=")[1])
		self.hidden_size=int(data.readline().split("=")[1])
		self.num_tokens=int(data.readline().split("=")[1])
		#print type(self.n_dict),self.n_pos,self.n_label,self.embedding_size,self.hidden_size,self.num_tokens
		self.known_words=[]
		self.known_poss=[]
		self.known_labels=[]
		sep=data.readline().strip().split()
		for i in range(n_dict):
			self.known_words.append(sep[i])
		sep=data.readline().strip().split()
		for i in range(n_pos):
			self.known_poss.append(sep[i])
		sep=data.readline().strip().split()
		for i in range(n_label):
			self.known_labels.append(sep[i])
		#print len(self.known_words),len(self.known_poss),len(self.known_labels)
		#load Eb
		Eb_size=n_dict+n_pos+n_label
		self.Eb=np.zeros([Eb_size,self.embedding_size])
		for i in range(Eb_size):
			sep=data.readline().strip().split()
			for j in range(self.embedding_size):
				self.Eb[i][j]=float(sep[j])
		#print self.Eb.shape
		#load W1
		sep=data.readline()
		self.W1=np.zeros([self.hidden_size,self.embedding_size*self.num_tokens])
		for i in range(self.hidden_size):
			sep=data.readline().strip().split()
			for j in range(self.embedding_size*self.num_tokens):
				self.W1[i][j]=float(sep[j])
		#print self.W1
		#load b1
		sep=data.readline()
		self.b1=np.zeros(self.hidden_size)
		sep=data.readline().strip().split()
		for i in range(self.hidden_size):
			self.b1[i]=sep[i]
		#print self.b1
		#load W2
		sep=data.readline()
		self.W2=np.zeros([2*n_label+1,self.hidden_size])
		for i in range(2*n_label+1):
			sep=data.readline().strip().split()
			for j in range(self.hidden_size):
				self.W2[i][j]=float(sep[j])
		#print self.W2
		data.close()

		#self.num_trans=2*n_label+1

		self.classifier=MLP.MLP([self.embedding_size*self.num_tokens,self.hidden_size,2*n_label+1],self.Eb,self.W1,self.b1,self.W2)


	def train(self,):
		sents=[]
		trees=[]
		self.load_file('en-universal-dev-brown.conll',sents,trees,True)#en-universal-dev-brown.conll
		self.print_tree_states(trees)

		(embed_ids,embeddings)=self.read_embed_file('embeds')#word_embeddings.txt
		(known_words,known_poss,known_labels)=self.gen_dictionaries(sents,trees,True)
		ldict=known_labels
		print ldict
		#ldict.remove('-NULL-')  #use -NULL- to denote the ROOT node's arc label
		self.system=ArcStandard(ldict,'CN',True)
		self.setup_classifier_for_trainning(sents,trees,True)

		self.classifier.train(3)
		self.save_model('model')

		self.test('test1')
		#test_tree=self.predict(sents[0])
		#test_tree.print_tree()
		#self.load_model('model')
		#c=Configuration(sents[0])
		#features=self.get_features(c)
		#scores=self.classifier.compute_scores(features)
		

	def test(self,test_filename):
		print "---loading test file from:",test_filename,"---"
		test_sents=[]
		test_trees=[]
		self.load_file(test_filename,test_sents,test_trees,True)#en-universal-dev-brown.conll

		#test_sents=sent
		#test_trees=tree

		self.print_tree_states(test_trees)
		n_sents=len(test_sents)
		predicted=[]
		self.system=ArcStandard(self.known_labels,'CN',True)
		for test_sent in test_sents:
			predicted.append(self.predict(test_sent))
		result=self.evaluate(test_sents,predicted,test_trees)
		print result

	def predict(self,sent):
		num_trans=len(self.system.transitions)
		c=Configuration(sent)
		while(not self.system.is_terminal(c)):
			opt_score=-10000
			opt_trans=''
			features=self.get_features(c)
			scores=self.classifier.compute_scores(features)
			#print scores
			for i in range(num_trans):
				if scores[i][0]>opt_score:
					#print "candidat:",self.system.transitions[i]
					if (self.system.can_apply(c,self.system.transitions[i])):
						opt_score=scores[i][0]
						opt_trans=self.system.transitions[i]
			#print "opt:",opt_trans
			#print c.info()
			#raw_in=raw_input("pause")
			self.system.apply(c,opt_trans)

		tree=c.tree
		return tree

	def evaluate(self,sents,predicted,gold_trees):
		sample_sum=0
		uas_cnt=0
		las_cnt=0
		for i in range(len(gold_trees)):
			sample_sum+=gold_trees[i].n
			for j in range(1,gold_trees[i].n+1):
				if predicted[i].get_head(j)==gold_trees[i].get_head(j):
					if predicted[i].get_head(j)==-1:
						print "error ! out of range!"
					elif predicted[i].get_label(j)==gold_trees[i].get_label(j):
						las_cnt+=1
					uas_cnt+=1
		result={}
		result['UAS']=uas_cnt/float(sample_sum)
		result['LAS']=las_cnt/float(sample_sum)
		return result

class Sample:
	def __init__(self,feature,label):
		self.feature=feature
		self.label=label

	def get_feature(self):
		return self.feature

	def get_label(self):
		return self.label

class Dataset:
	def __init__(self,num_features,num_labels):
		self.n=0
		self.num_features=num_features
		self.num_labels=num_labels
		self.samples=[]

	def add_sample(self,feature,label):
		sample=Sample(feature,label)
		self.n+=1
		self.samples.append(sample)

	def print_info(self):
		print self.n
		for i in range(self.n):
			print self.samples[i].get_feature()," ",self.samples[i].get_label()

	def shuffle(self):
		np.random.shuffle(self.samples)

class Configuration:
	def __init__(self,sent):
		self.stack=[]
		self.buffer=[]
		self.sent=sent
		self.tree=DependencyTree()
		for i in range(1,self.sent.n+1):
			self.tree.add(-1,'-UNKNOWN-')
			self.buffer.append(i)
		self.stack.append(0)

	def shift(self):
		k=self.get_buffer(0)
		if k==-1:
			return False
		self.buffer.pop(0)
		self.stack.append(k)
		return True

	def remove_top_stack(self):
		n_stack=self.get_stack_size()
		if n_stack<1:
			return False
		self.stack.pop()
		return True

	def remove_second_top_stack(self):
		n_stack=self.get_stack_size()
		if n_stack<2:
			return False
		self.stack.pop(-2)
		return True

	def get_stack_size(self):
		return len(self.stack)

	def get_buffer_size(self):
		return len(self.buffer)

	def get_sent_size(self):
		return len(self.sent.n)

	def get_head(self,k):
		return self.tree.get_head(k)

	def get_label(self,k):
		return self.tree.get_label(k)

	#k starts from 0,top stack
	def get_stack(self,k):
		n_stack=self.get_stack_size()
		if (k>=0 and k<n_stack):
			return self.stack[-1-k]
		else:
			return -1

	def get_buffer(self,k):
		n_buffer=self.get_buffer_size()
		if (k>=0 and k<n_buffer):
			return self.buffer[k]
		else:
			return -1

	def get_word(self,k):
		if (k==0):
			return '-ROOT-'
		else:
			k=k-1
		if (k<0 or k>=self.sent.n):
			return 'NULL'
		else:
			return self.sent.words[k]

	def get_pos(self,k):
		if (k==0):
			return '-ROOT-'
		else:
			k=k-1
		if (k<0 or k>=self.sent.n):
			return 'NULL'
		else:
			return self.sent.poss[k]

	#m's parent is h, label is l,the arc is from h to m
	def add_arc(self,h,m,l):
		self.tree.set(m,h,l)

	def get_left_child(self,k,cnt):
		if (k<0 or k>self.tree.n):
			return -1
		c=0
		for i in range(1,k):
			if self.tree.get_head(i)==k:
				c+=1
				if c==cnt:
					return i
		return -1

	#def get_left_child(self,k):
		#return self.get_left_child(k,1)

	def get_right_child(self,k,cnt):
		if (k<0 or k>self.tree.n):
			return -1
		c=0
		for i in range(self.tree.n,k,-1):
			if self.tree.get_head(i)==k:
				c+=1
				if c==cnt:
					return i
		return -1

	#def get_right_child(self,k):
		#return self.get_right_child(k,1)

	def has_other_child(self,k,gold_tree):
		for i in range(1,self.tree.n+1):
			if (gold_tree.get_head(i)==k and self.tree.get_head(i)!=k):
				return True
		return False

	def get_tree(self):
		return self.tree

	def info(self):
		s="[S]"
		for i in range(self.get_stack_size()):
			if i>0:
				s+=","
			s+=str(self.stack[i])

		s+="\n[B]"
		for i in range(self.get_buffer_size()):
			if i>0:
				s+=","
			s+=str(self.buffer[i])

		s+="\n[H]"
		for i in range(1,self.tree.n+1):
			if i>1:
				s+=","
			s+=str(self.get_head(i))+"("+str(self.get_label(i))+")"
		return s

	

if __name__=="__main__":
	#arc=ArcStandard()
	parser=Parser()
	parser.train()
	"""
	print known_words
	print known_poss
	print known_labels
	sents[0].print_info()
	trees[0].print_tree()
	print trees[0].is_single_root()
	print trees[0].is_projective()
	"""


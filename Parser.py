#!/usr/bin/python
#coding=utf-8
import numpy as np
import copy
import MLP
import os
import ArcStandard,DependencySent,DependencyTree,Dataset,Configuration
import Config
"""
import DependencySent
import DependencyTree
import Dataset
import Configuration
"""

class Parser:
	def __init__(self):
		self.word_ids={}
		self.label_ids={}
		self.pos_ids={}
		self.pre_computed_ids={}
		self.config=Config.Config()
		self.delexicalized=self.config.delexicalized
		self.use_postag=self.config.use_postag
		self.labeled=self.config.labeled
		self.num_tokens=self.config.num_tokens
		self.embedding_size=self.config.embedding_size
		self.num_pre_computed=self.config.num_pre_computed
		self.hidden_size=self.config.hidden_size
		self.pos_emb_size=self.config.pos_emb_size
		self.label_emb_size=self.config.label_emb_size

	def train(self,):
		sents=[]
		trees=[]
		self.load_file(self.config.training_file_name,sents,trees,True)#en-universal-dev-brown.conll#en-universal-train-brown.conll
		self.print_tree_states(trees)

		(embed_ids,embeddings)=self.read_embed_file(self.config.embedding_file_name)#word_embeddings.txt
		(known_words,known_poss,known_labels)=self.gen_dictionaries(sents,trees,True)
		ldict=known_labels
		#print ldict
		#ldict.remove('-NULL-')  #use -NULL- to denote the ROOT node's arc label
		self.system=ArcStandard.ArcStandard(ldict,'CN',True)
		self.setup_classifier_for_trainning(sents,trees,True)

		self.classifier.train(10)
		
		#test_tree=self.predict(sents[0])
		#test_tree.print_tree()
		#self.load_model('model')
		#c=Configuration(sents[0])
		#features=self.get_features(c)
		#scores=self.classifier.compute_scores(features)

	def load_file(self,file,sents,trees,labeled):
		sent=DependencySent.DependencySent()
		tree=DependencyTree.DependencyTree()
		data=open(file)
		for line in data:
			sep_line=line.strip().split()
			#print sep_line
			if len(sep_line)<10:
				sents.append(sent)
				trees.append(tree)
				sent=DependencySent.DependencySent()
				tree=DependencyTree.DependencyTree()
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
		print "---tree states---"
		print "there are ",len(trees),"trees"
		non_trees=0
		non_proj=0
		for i in range(len(trees)):
			if not trees[i].is_tree():
				non_trees+=1
			elif not trees[i].is_projective():
				non_proj+=1
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
		print "creating classifier (",self.embedding_size*self.num_tokens,",",self.hidden_size,",",n_actions,")"
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
		ds_train=Dataset.Dataset(num_tokens,num_trans)
		print "---generating training examples---"
		tokpos_count={}
		for i in range(len(sents)):
			if i%1000==0:
				print "have processed",i,"sents"
		#only use projective tree
			if trees[i].is_projective():
				c=Configuration.Configuration(sents[i])
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
		print "##train examples num:",ds_train.n,"##"
		temp=copy.deepcopy(tokpos_count)
		#print "sort tokpos_count"
		temp=sorted(temp.iteritems(),key=lambda d:d[1],reverse=True)
		self.pre_computed_ids=[]
		#print "fill pre_computed_ids"
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
		self.generate_ids()
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
		self.system=ArcStandard.ArcStandard(self.known_labels,'CN',True)
		for test_sent in test_sents:
			predicted.append(self.predict(test_sent))
		result=self.evaluate(test_sents,predicted,test_trees)
		print result

	def predict(self,sent):
		num_trans=len(self.system.transitions)
		c=Configuration.Configuration(sent)
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

if __name__=="__main__":
	#arc=ArcStandard()
	parser=Parser()
	if not parser.config.is_test:
		parser.train()
		self.save_model(parser.config.save_model_name)
		self.test(parser.config.test_file_name)#en-universal-dev-brown.conll
	else:
		parser.load_model(parser.config.load_file_name)
		parser.test(parser.config.test_file_name)

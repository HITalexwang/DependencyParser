import DependencyTree

class Configuration:
	def __init__(self,sent):
		self.stack=[]
		self.buffer=[]
		self.sent=sent
		self.tree=DependencyTree.DependencyTree()
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
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
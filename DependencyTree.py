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

	def get_projective_order(self):
		self.projective_order={}
		self.pro_order=0
		self.inorder_traversal(0)
		return self.projective_order
		
	def inorder_traversal(self,n):
		for i in range(1,n):
			if self.get_head(i)==n:
				self.inorder_traversal(i)
		self.projective_order[n]=self.pro_order
		self.pro_order+=1
		for i in range(n+1,self.n+1):
			if self.get_head(i)==n:
				self.inorder_traversal(i)

	def get_mpc(self):
		self.mpc=[-1]
		self.comp_num=0
		for i in  range(1,self.n+1):
			self.mpc.append(-1)
		self.visit_node(0)
		return self.mpc

	def visit_node(self,n):
		split_flag=False
		left=n
		right=n
		for i in range(n-1,0,-1):
			if self.get_head(i)==n:
				child=self.visit_node(i)
				# if get None from child, pass None to parent, BUT continue merge child of the node itself
				if child is None:
					split_flag=True
				else:
					(l,r)=child
					#merge the child to this node
					if right==l-1:
						right=r
					elif left==r+1:
						left=l
					# if can not merge(not in neighborhood),plot the childs on the list,pass None to parent
					else:
						split_flag=True
						self.comp_num+=1
						for i in range(l,r+1):
							self.mpc[i]=self.comp_num
		for i in range(n+1,self.n+1):
			if self.get_head(i)==n:
				child=self.visit_node(i)
				if child is None:
					split_flag=True
				else:
					(l,r)=child
					if right==l-1:
						right=r
					elif left==r+1:
						left=l
					else:
						split_flag=True
						self.comp_num+=1
						for i in range(l,r+1):
							self.mpc[i]=self.comp_num
		#if the child returns none or there is a split node in childs,plot the combined childs of this node 
		#and return None to parent to convey the message that there exists a split node on the path
		if split_flag==True:
			self.comp_num+=1
			for i in range(left,right+1):
				self.mpc[i]=self.comp_num
			return None
		else:
			return (left,right)

	def print_tree(self):
		for i in range(0,self.n+1):
			print i,self.get_head(i),',',self.get_label(i)

"""
	def visit_node(self,n):
		child_comp=[]
		for i in range(n-1,0,-1):
			if self.get_head(i)==n:
				child_comp.extend(self.visit_node(i))
		for i in range(n+1,self.n+1):
			if self.get_head(i)==n:
				child_comp.extend(self.visit_node(i))
		if n==0:
			self.mpc[0]=0
			child_comp.append(0)
			return child_comp
		elif n==self.n:
			self.comp_num+=1
			self.mpc[n]=self.comp_num
			child_comp.append(self.comp_num)
			return child_comp
		elif not self.mpc[n-1]==self.comp_num and not self.mpc[n+1]==self.comp_num:
			for comp in child_comp:
				if self.mpc[n-1]==comp or self.mpc[n+1]==comp:
					self.mpc[n]=comp
					return []
			self.comp_num+=1
			self.mpc[n]=self.comp_num
			child_comp.append(self.comp_num)
			return child_comp
		else:
			self.mpc[n]=self.comp_num
			return child_comp"""
			
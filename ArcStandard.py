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
		self.transitions.append('W')#denote swap
		"""
		print "transition types:"
		for i in range(len(self.transitions)):
			print selfself.transitions[i]
		"""
	def can_apply(self,c,t):#configuration c,transition t
		if (t.startswith('R') or t.startswith('L')):
			label=t[2:-1]
			if t.startswith("L"):
				h=c.get_stack(0)#stack top
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
		elif t=="W":
			return n_stack>2
		else: #shift
			#print "can apply S",n_buffer
			return n_buffer>0

	def apply(self,c,t):#configuration c
		w1=c.get_stack(1)#second
		w2=c.get_stack(0)#top
		#left
		if t.startswith("L"):
			c.add_arc(w2,w1,t[2:-1])
			c.remove_second_top_stack()
			#c.lvalency[w2] += 1;
		#right
		elif t.startswith("R"):
			c.add_arc(w1,w2,t[2:-1])
			c.remove_top_stack()
		elif t=="W":
			c.swap()
		else:
			c.shift()

	def get_oracle(self,c,tree):#configuration c
		w1=c.get_stack(1)#second
		w2=c.get_stack(0)#top
		projective_flag=tree.is_projective()
		if not projective_flag:
			#projective_order=tree.get_projective_order()
			#mpc=tree.get_mpc()
			b1=c.get_buffer(0)#next to move in
		if (w1>0 and tree.get_head(w1)==w2 and (not c.has_other_child(w1,tree))):
			return "L("+tree.get_label(w1)+")"
		elif (w1>=0 and tree.get_head(w2)==w1 and (not c.has_other_child(w2,tree))):
			return "R("+tree.get_label(w2)+")"
		elif not projective_flag:
			#special case when buffer is empty,predict swap when projective order is reverse
			if w1>0 and b1==-1 and tree.projective_order[w1]>tree.projective_order[w2]:
				return "W"
			elif w1>0 and b1>0 and tree.projective_order[w1]>tree.projective_order[w2] and not tree.mpc[w2]==tree.mpc[b1]:
				return "W"
			else:
				return "S"
		else:
			return "S"

	def init_configuration(self,sent):
		c=Configuration(sent)
		return c

	def is_terminal(self,c):#configuration c
		return (c.get_buffer_size()==0 and c.get_stack_size()==1)
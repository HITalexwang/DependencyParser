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

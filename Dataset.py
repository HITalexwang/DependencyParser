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
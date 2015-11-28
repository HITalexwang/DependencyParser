class Config:
	def __init__(self):
		self.delexicalized=False
		self.use_postag=True
		self.labeled=True
		self.num_tokens=48
		self.embedding_size=50
		self.num_pre_computed=100000
		self.hidden_size=100
		self.pos_emb_size=10
		self.label_emb_size=10

        	self.reg_parameter=1.0e-8
        	self.batch_size=10000
        	self.alpha=0.01
        	self.ada_eps=1.0e-6
        	self.training_threads=10

        	self.training_file_name='data/en-universal-dev-brown.conll'
        	self.embedding_file_name='data/word_embeddings.txt'
        	self.test_file_name='data/en-universal-dev-brown.conll'
        	self.save_model_name='data/h_100,E_50_model'


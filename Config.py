class Config:
	def __init__(self):
		self.check=True
		self.is_test=False
		
		self.iter=15
		self.checkiter=5
		self.out_iter=self.iter/self.checkiter

		self.delexicalized=False
		self.use_postag=True
		self.labeled=True
		self.num_tokens=48
		self.embedding_size=5
		self.pos_emb_size=5
		self.label_emb_size=5
		self.num_pre_computed=100000
		self.hidden_size=2
		self.word_tokens_num=18
		self.pos_tokens_num=18
		self.label_tokens_num=12
		self.pos_tokens_up_bound=self.word_tokens_num+self.pos_tokens_num#18+18
		self.label_tokens_up_bound=self.pos_tokens_up_bound+self.label_tokens_num#18+18+12
		self.input_length=self.word_tokens_num*self.embedding_size+self.pos_tokens_num*self.pos_emb_size+self.label_tokens_num*self.label_emb_size

        	self.reg_parameter=1.0e-8
        	self.batch_size=10000
        	self.alpha=0.01
        	self.ada_eps=1.0e-6
        	self.training_threads=6
        	self.pre_threads=1

        	self.training_file_name='data/samples/test1'#en-universal-dev-brown.conll
        	self.embedding_file_name='data/embeddings/en.50'#word_embeddings.txt
        	self.test_file_name='data/samples/test1'#en-universal-dev-brown.conll
        	self.save_model_name='data/models/test_model'
        	self.load_file_name='data/models/test_model0'


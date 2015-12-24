import DependencyTree,Parser,ArcStandard,Dataset

def print_tree_states(trees,sents):
	print "---tree states---"
	print "there are ",len(trees),"trees"
	non_trees=0
	non_proj=0
	non_file=open('data/samples/non-proj-sents.txt','w')
	for i in range(len(trees)):
		if not trees[i].is_tree():
			non_trees+=1
		elif not trees[i].is_projective():
			non_proj+=1
			for j in range(1,trees[i].n+1):
				non_file.write(str(j)+"\t"+sents[i].words[j-1]+"\t"+sents[i].words[j-1]+"\t"+sents[i].poss[j-1]
					+"\t"+sents[i].poss[j-1]+"\t_\t"+str(trees[i].get_head(j))+"\t"+trees[i].get_label(j)+"\t_\t_\n")
			non_file.write("\n")
	print non_trees,"trees are illegal"
	print non_proj,"trees are not projective"

parser=Parser.Parser()
sents=[]
trees=[]
parser.load_file('data/samples/sdpv2.train.cor.conll',sents,trees,True)#sdpv2.train.cor.conll
parser.print_tree_states(trees,sents)
"""
(known_words,known_poss,known_labels)=parser.gen_dictionaries(sents,trees,True)
ldict=known_labels
system=ArcStandard.ArcStandard(ldict,'CN',True)
print len(system.transitions)
#dataset=parser.gen_train_samples(sents,trees)
"""

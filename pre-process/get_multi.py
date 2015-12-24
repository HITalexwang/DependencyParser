def read(file):
	data=open(file)
	multi_head_flag=False
	multi_head_cnt=0
	sent=""
	mul_out=open("data/samples/non-proj/sdpv2.dev.trees.conll","w")
	cnt=0
	for line in data:
		sep_line=line.strip().split()
		if len(sep_line)<10:
			if multi_head_flag:
				multi_head_cnt+=1
				mul_out.write(sent+"\n")
			sent=""
			cnt=0
			multi_head_flag=False
		else:
			cnt+=1
			num=int(sep_line[0])
			sent+=line
			if num!=cnt:
				if not multi_head_flag:
					multi_head_flag=True
	data.close()
	print "get",multi_head_cnt,"sents"

read("data/samples/non-proj/sdpv2.dev.cor.conll",True)
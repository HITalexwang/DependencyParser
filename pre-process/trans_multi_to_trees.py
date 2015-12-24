def read(file,write_flag):
	data=open(file)
	multi_head_flag=False
	multi_head_cnt=0
	sent=""
	if write_flag:
		mul_out=open("../data/samples/non-proj/sdpv2.test.trees.conll","w")
	cnt=0
	total=0
	for line in data:
		sep_line=line.strip().split()
		if len(sep_line)<10:
			if multi_head_flag:
				multi_head_cnt+=1
			if write_flag:
				mul_out.write(sent+"\n")
			sent=""
			total+=1
			cnt=0
			multi_head_flag=False
		else:
			cnt+=1
			num=int(sep_line[0])
			if num==cnt:
				sent+=line
			else:
				cnt-=1
				if not multi_head_flag:
					multi_head_flag=True
	data.close()
	print "get",multi_head_cnt,"multi head sents out of ",total,"sents"

read("../data/samples/non-proj/sdpv2.test.cor.conll",True)#sdpv2.train.cor.conll
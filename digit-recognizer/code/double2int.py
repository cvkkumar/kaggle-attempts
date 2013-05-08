
f = open ("result.txt")
fout = open("./omresult.txt","w")
for line in f:
	n = float(line.strip());
	x = str(int(n)) 
	print n,x
	fout.write(x+"\n");
f.close()
fout.close()


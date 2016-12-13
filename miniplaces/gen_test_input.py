with open("development_kit/data/test.txt", "r") as f:
	output = ""
	for l in f.readlines():
		s = l.split(" ")[0]
		n = str(int(s.split("/")[1].split(".")[0]))
		output += " ".join([s,n])+"\n"
	with open("development_kit/data/test_new.txt","w") as o:
		o.write(output)


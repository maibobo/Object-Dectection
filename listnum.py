import os

f = open("num.txt", "a+")
dir="D:/ship/ship_name/test_oldversion/LocResult/SHIPNAME_IMG"
files=os.listdir(dir)
for file in files:
	fi_d = os.path.join(dir,file)  
	subfiles=os.listdir(fi_d)
	outline='%s,%d\n'%(file,len(subfiles))
	f.write(outline)

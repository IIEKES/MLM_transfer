import sys
import random
import string
import json
import os
operation=sys.argv[3]
dict_num=string.atof(sys.argv[4])
dict_thre=string.atof(sys.argv[5])

def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

save_path = os.path.join(os.curdir, sys.argv[8], sys.argv[7])
if not os.path.exists(save_path):
	os.mkdir(save_path)

frname = os.path.join(save_path, sys.argv[2]+'.tf_idf'+'.'+operation)
f=open(frname,'r')
word_dict={}
num=0
for line in f:
	try:
		lines=line.strip().decode('utf-8').encode('gb18030').split('\t')
	except:
		continue
	if(len(lines)!=2):
		continue
	if(string.atof(lines[1])>dict_thre and num<dict_num):
		word_dict[lines[0]]=1
		num+=1
f.close()

'''
frname2 = os.path.join(os.curdir, sys.argv[8], sys.argv[7], sys.argv[2]+'.tf_idf'+'.cls.clean.'+operation)
f2=open(frname2,'r')
word_dict2={}
num=0
for line in f2:
	try:
		lines=line.strip().decode('utf-8').encode('gb18030').split('\t')
	except:
		continue
	if(len(lines)!=2):
		continue
	if(string.atof(lines[1])>0.6 and num<dict_num):
		word_dict2[lines[0]]=1
		num+=1
f2.close()
'''
frname = os.path.join(save_path, sys.argv[1])
f=open(sys.argv[1],'r')
fwname = os.path.join(save_path, sys.argv[6]+'.data.'+operation)
fw=open(fwname,'w')
fwname2 = os.path.join(save_path, sys.argv[6]+'.unable.'+operation)
fw2=open(fwname2,'w')

processed_num =0
for line in f:
	try:
		lines=line.strip().decode('utf-8').encode('gb18030').split(' ')
		lines = ' '.join(lines).split() # Add by xing to remove extra space within sentence
	except:
		continue
	ll = len(lines)
	lim = min(4, ll / 2)
	contents=[]
	n_gram = 4
	while (len(contents) < 5):
		content=''
		style_dict=[]
		for i in range(len(lines)):
			for n in range(n_gram,0,-1):
				if(i+n>len(lines)):
					continue
				if(word_dict.get(' '.join(lines[i:i+n]))!=None and (style_dict==[] or i+n-1 >style_dict[-1])):
					style_dict.append(i)
					style_dict.append(i+n-1)
					break
		start=0
		if(len(style_dict)>0 and style_dict[0]==0):
			content=''
		masks = []
		for i in range(0,len(style_dict),2):
			if (start < style_dict[i]):
				content += ' '.join(lines[start:style_dict[i]]) + ' '
			masks.extend([m for m in range(style_dict[i], style_dict[i+1]+1)])
			start=style_dict[i+1]+1
		if (start < len(lines)):
			content += ' '.join(lines[start:len(lines)]) + ' '
		content=content.strip()
		contents=content.strip().split(' ')
		n_gram -= 1
		if n_gram == 0:
			break

	if (len(contents) < 5):
		#print("++:{}".format(line))
		fw2.write(line)
		continue
	masks = list(set(masks))
	if masks!=[]:
		processed_num += 1
		masks.sort()
		if operation=='label':
			style=sys.argv[1][-1]
			wl = {"content":content, "line":line.strip(), "masks":masks, "label": style}
			#fw.write(content+'\t'+ line.strip()+'\t'+masks+'\n')
			wl_str = json.dumps(wl)
			fw.write(wl_str)
			fw.write("\n")
	else:
		fw2.write(line)
f.close()
fw.close()
fw2.close()
print("processed_num={}".format(processed_num))

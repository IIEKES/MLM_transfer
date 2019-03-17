import sys
import random
import string
import os
import json
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
words_dict=[]
word_dict={}
num=0
for line in f:
	try:
		lines=line.strip().decode('utf-8').encode('gb18030').split('\t')
	except:
		continue
	#print len(lines)
	#print line
	if(len(lines)!=2):
		continue
	if(string.atof(lines[1])>dict_thre and num<dict_num):
		word_dict[lines[0]]=1
		num+=1
f.close()
for i in word_dict.keys():
	words=i.split(' ')
	words_dict.append(words)
#print len(word_dict)
f=open(sys.argv[1],'r')
fwname = os.path.join(os.curdir, sys.argv[8], sys.argv[7], sys.argv[6]+'.data.'+operation)
fw=open(fwname,'w')

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

	content = ''
	style_dict = []
	for i in range(len(lines)):
		for n in range(n_gram, 0, -1):
			if (i + n > len(lines)):
				continue
			if (word_dict.get(' '.join(lines[i:i + n])) != None and (style_dict == [] or i + n - 1 > style_dict[-1])):
				style_dict.append(i)
				style_dict.append(i + n - 1)
				break
	start = 0
	if (len(style_dict) > 0 and style_dict[0] == 0):
		content = ''
	masks = []
	for i in range(0, len(style_dict), 2):
		if (start < style_dict[i]):
			content += ' '.join(lines[start:style_dict[i]]) + ' '
		masks.extend([m for m in range(style_dict[i], style_dict[i + 1] + 1)])
		start = style_dict[i + 1] + 1
	if (start < len(lines)):
		content += ' '.join(lines[start:len(lines)]) + ' '
	content = content.strip()
	contents = content.strip().split(' ')
	n_gram -= 1
	if n_gram == 0:
		break
	masks = list(set(masks))

	if masks != [] and  contents == []:
		n = max(len(lines)/2, 1)
		rand = random.sample(range(0,len(lines)),n)
		for r in rand:
			contents.append(lines[r])
			del masks[rand]
		content = ' '.join(contents[:])
		style = ' '.join(lines[:])
	if masks == []:
		rand = random.randint(0,len(lines)-1)
		masks.append(rand)
		del lines[rand]
		content = ' '.join(lines[:])

	masks.sort()
	style=sys.argv[1][-1]
	wl = {"content":content, "line":line.strip(), "masks":masks, "label": style}
	#fw.write(content+'\t'+ line.strip()+'\t'+masks+'\n')
	wl_str = json.dumps(wl)
	fw.write(wl_str)
	fw.write("\n")
	processed_num += 1

f.close()
fw.close()
print("processed_num={}".format(processed_num))


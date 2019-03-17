import json
import sys

num=sys.argv[1]
model_name=sys.argv[2]
task_name=sys.argv[3]

fs = open("processed_data/{}/{}.test.{}.data.label".format(task_name, task_name, num), "r")
can = []
for l in fs:
    line = json.loads(l).get('line')
    can.append(line.strip())
fs.close()
print(len(can))

fl = open("evaluation/outputs/{}/sentiment.test.{}.{}".format(task_name, num, model_name), "r")
fw = open("evaluation/outputs/{}/sentiment.test.{}.{}.s".format(task_name, num, model_name), "w")
for l in fl:
    line=l.strip().split('\t')
    if (len(line) == 3):
        if line[0] in can:
            fw.write(l)
fl.close()
fw.close()
import os
import json
import numpy
import matplotlib
import matplotlib.pyplot as plt

log_path = "."
files = [
#"screenlog_fine_tune_cbert_w_cls_amazon.log",
#"screenlog_fine_tune_cbert_w_cls_amazon_modified.log",
"screenlog_fine_tune_yelp_fusion_method.log",
#"screenlog_fine_tune_cbert_w_cls_pg_yelp.log",
#"screenlog_fine_tune_cbert_w_cls_pg_yelp_modified.log",
"screenlog_fine_tune_amazon_fusion_method.log"
]
titles = [
    "Dataset:Yelp",
    "Dataset:Amazon"
]

fig = plt.figure(1)
plt.subplots_adjust(wspace =0.5, hspace =0.5)

for fi, file in enumerate(files):
    print("polting {}".format(file))
    filename = os.path.join(log_path, file)
    fr = open(filename, "r")

    point = dict()
    lines = fr.readlines()
    i = 0
    while i < len(lines):
        #print(i)
        if "bleu_avg" not in lines[i]:
            i += 1
            continue
        js = json.loads(lines[i])
        bleu = js["bleu_avg"]
        i += 1
        js = json.loads(lines[i])
        acc = js["acc_avg"]
        i += 1
        if acc not in point.keys():
            point[acc] = bleu
        elif bleu > point[acc]:
            point[acc] = float(bleu)

    x = []
    for p in point.items():
        x.append([p[0], p[1]])
    x = numpy.stack(x)
    ax1 = fig.add_subplot(2,2,fi+1)
    ax1.set_xlabel('accuracy')
    ax1.set_ylabel('BLEU')
    ax1.set_title(titles[fi])
    plt.scatter(x[:,0],x[:,1])
print("done")
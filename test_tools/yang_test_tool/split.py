import sys
import os
import glob

def run_split(f):
    fr = open(f, 'r')
    fname = "{}.split".format(f)
    fw = open(fname, 'w')
    for line in fr:
        splits = line.strip().split('\t')
        if len(splits) == 3:
            fw.write(splits[1])
            fw.write('\n')
    fr.close()
    fw.close()


if __name__ == "__main__":
    eval_dir = "evaluation/outputs"
    files = glob.glob(os.path.join(eval_dir, "yelp/*"))

    for f in files:
        run_split(f)
import os

def eval_multi_bleu(model_name, task_name):
    eval_path = "evaluation/outputs/{}".format(task_name)
    multi_bleu_perl = "test_tools/yang_test_tool/{}".format("multi-bleu.perl")
    r = os.popen("perl {} {}/sentiment.test.0.human.split < {}/sentiment.test.0.{}.split".format(multi_bleu_perl, eval_path, eval_path, model_name))
    info = r.readlines()
    bleu_0 = float(info[0].split(',')[0].split('=')[1])
    r = os.popen("perl {} {}/sentiment.test.1.human.split < {}/sentiment.test.1.{}.split".format(multi_bleu_perl, eval_path, eval_path, model_name))
    info = r.readlines()
    bleu_1 = float(info[0].split(',')[0].split('=')[1])
    bleu = (bleu_0 + bleu_1) / 2
    return bleu


if __name__ == "__main__":
    print("bleu:{}". format(eval_multi_bleu("mit", "yelp")))

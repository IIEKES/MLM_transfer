results = [
[[74.7,2.5],[69.1,0.9],[73.7,3.1],[74.1,0.4]],
[[8.1,12.9],[31.2,11.3],[8.7,11.8],[43.3,10.0]],
[[54.1,6.7],[62.0,5.2],[47.6,7.1],[68.3,5.0]],
[[87.2,14.1],[61.4,24.2],[81.7,11.8],[68.7,27.1]],
[[95.4,0.5],[69.7,0.9],[95.4,0.4],[70.3,0.9]],
[[89.8,9.5],[46.0,21.8],[85.7,7.5],[45.6,24.6]],
[[93.1,10.5],[50.5,19.6],[88.7,8.4],[48.0,22.8]],
[[41.3,16.1],[37.3,27.9],[58.9,14.6],[64.2,27.9]],
[[93.7,14.7],[65.2,23.0],[96.6,12.7],[88.0,25.8]],
[[95.9,14.8],[66.4,24.6],[96.7,13.7],[87.9,27.3]]
]

models = [
"CROSSALIGNED",
"STYLEEMBEDDING",
"MULTIDECODER",
"TEMPLATEBASED",
"RETRIEVEONLY",
"DELETEONLY",
"DELETEANDRETRIEVE",
"LC-MLM",
"LC-MLM-PG",
"LC-MLM-SS"
]

for model, result in zip(models,results):
    line = model
    for acc, bleu in result:
        fs = bleu * acc * 2 / (bleu + acc)
        line =  line + '& ' + str(round(acc,2)) + '& ' + str(round(bleu,2)) + '& ' + str(round(fs,2))
    line = line + '\\\\'
    print(line)

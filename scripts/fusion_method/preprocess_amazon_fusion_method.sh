set -x
cp configs/cbert_amazon_fusion_method.config run.config
PROCESSED_DATA_DIR=processed_data_fusion_method
#/home/xgg/anaconda3/envs/py27/bin/python2.7 filter_style_ngrams_modified.py raw_data/amazon/sentiment.train. 2 label amazon.train. amazon $PROCESSED_DATA_DIR
#/home/xgg/anaconda3/envs/py27/bin/python2.7 use_nltk_to_filter.py $PROCESSED_DATA_DIR/amazon/amazon.train.0.tf_idf.filter.label
#cp $PROCESSED_DATA_DIR/amazon/amazon.train.0.tf_idf.filter.label.filter $PROCESSED_DATA_DIR/amazon/amazon.train.0.tf_idf.filter.label
#/home/xgg/anaconda3/envs/py27/bin/python2.7 use_nltk_to_filter.py $PROCESSED_DATA_DIR/amazon/amazon.train.1.tf_idf.filter.label
#cp $PROCESSED_DATA_DIR/amazon/amazon.train.1.tf_idf.filter.label.filter $PROCESSED_DATA_DIR/amazon/amazon.train.1.tf_idf.filter.label
cp drg_tf_idf/amazon/sentiment.train.0.tf_idf.orgin.filter $PROCESSED_DATA_DIR/amazon/amazon.train.0.tf_idf.filter.label
cp drg_tf_idf/amazon/sentiment.train.1.tf_idf.orgin.filter $PROCESSED_DATA_DIR/amazon/amazon.train.1.tf_idf.filter.label
/home/xgg/anaconda3/bin/python attn_cls_tf_idf.py $PROCESSED_DATA_DIR
cp $PROCESSED_DATA_DIR/amazon/amazon.train.0.tf_idf.attn.label $PROCESSED_DATA_DIR/amazon/amazon.train.0.tf_idf.label
cp $PROCESSED_DATA_DIR/amazon/amazon.train.1.tf_idf.attn.label $PROCESSED_DATA_DIR/amazon/amazon.train.1.tf_idf.label
/home/xgg/anaconda3/envs/py27/bin/python2.7 preprocess_fusion_method.py raw_data/amazon/sentiment.train.0 amazon.train.0 label 10000 1 amazon.train.0 amazon $PROCESSED_DATA_DIR
/home/xgg/anaconda3/envs/py27/bin/python2.7 preprocess_fusion_method.py raw_data/amazon/sentiment.dev.0 amazon.train.0 label 10000 1 amazon.dev.0 amazon $PROCESSED_DATA_DIR
python preprocess_fusion_method_for_test.py raw_data/amazon/sentiment.test.0 amazon.train.0 label 10000 1 amazon.test.0 amazon $PROCESSED_DATA_DIR
/home/xgg/anaconda3/envs/py27/bin/python2.7 preprocess_fusion_method.py raw_data/amazon/sentiment.train.1 amazon.train.1 label 10000 1 amazon.train.1 amazon $PROCESSED_DATA_DIR
/home/xgg/anaconda3/envs/py27/bin/python2.7 preprocess_fusion_method.py raw_data/amazon/sentiment.dev.1 amazon.train.1 label 10000 1 amazon.dev.1 amazon $PROCESSED_DATA_DIR
python preprocess_fusion_method_for_test.py raw_data/amazon/sentiment.test.1 amazon.train.1 label 10000 1 amazon.test.1 amazon $PROCESSED_DATA_DIR
/home/xgg/anaconda3/bin/python process_unable.py $PROCESSED_DATA_DIR
rm $PROCESSED_DATA_DIR/amazon/train.data.label
cat $PROCESSED_DATA_DIR/amazon/amazon.train.*.data.label >> $PROCESSED_DATA_DIR/amazon/train.data.label
rm $PROCESSED_DATA_DIR/amazon/dev.data.label
cat $PROCESSED_DATA_DIR/amazon/amazon.dev.*.data.label >> $PROCESSED_DATA_DIR/amazon/dev.data.label
rm $PROCESSED_DATA_DIR/amazon/test.data.label
cat $PROCESSED_DATA_DIR/amazon/amazon.test.*.data.label >> $PROCESSED_DATA_DIR/amazon/test.data.label
/home/xgg/anaconda3/envs/py27/bin/python2.7 shuffle.py $PROCESSED_DATA_DIR/amazon/train.data.label
/home/xgg/anaconda3/envs/py27/bin/python2.7 shuffle.py $PROCESSED_DATA_DIR/amazon/dev.data.label
cp $PROCESSED_DATA_DIR/amazon/train.data.label.shuffle $PROCESSED_DATA_DIR/amazon/train.data.label
cp $PROCESSED_DATA_DIR/amazon/dev.data.label.shuffle $PROCESSED_DATA_DIR/amazon/dev.data.label

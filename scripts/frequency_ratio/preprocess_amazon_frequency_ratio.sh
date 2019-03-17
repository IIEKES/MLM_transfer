set -x
cp configs/cbert_amazon_frequency_ratio.config run.config
PROCESSED_DATA_DIR=processed_data_frequency_ratio
#/home/xgg/anaconda3/envs/py27/bin/python2.7 filter_style_ngrams.py raw_data/amazon/sentiment.train. 2 label amazon.train. amazon $PROCESSED_DATA_DIR
#/home/xgg/anaconda3/envs/py27/bin/python2.7 use_nltk_to_filter.py $PROCESSED_DATA_DIR/amazon/amazon.train.0.tf_idf.label
#cp $PROCESSED_DATA_DIR/amazon/amazon.train.0.tf_idf.label.filter $PROCESSED_DATA_DIR/amazon/amazon.train.0.tf_idf.label
#/home/xgg/anaconda3/envs/py27/bin/python2.7 use_nltk_to_filter.py $PROCESSED_DATA_DIR/amazon/amazon.train.1.tf_idf.label
#cp $PROCESSED_DATA_DIR/amazon/amazon.train.1.tf_idf.label.filter $PROCESSED_DATA_DIR/amazon/amazon.train.1.tf_idf.label
cp drg_tf_idf/amazon/sentiment.train.0.tf_idf.orgin $PROCESSED_DATA_DIR/amazon/amazon.train.0.tf_idf.label
cp drg_tf_idf/amazon/sentiment.train.1.tf_idf.orgin $PROCESSED_DATA_DIR/amazon/amazon.train.1.tf_idf.label
/home/xgg/anaconda3/envs/py27/bin/python2.7 preprocess_train.py raw_data/amazon/sentiment.train.0 amazon.train.0 label 10000 1 amazon.train.0 amazon $PROCESSED_DATA_DIR
/home/xgg/anaconda3/envs/py27/bin/python2.7 preprocess_train.py raw_data/amazon/sentiment.dev.0 amazon.train.0 label 10000 1 amazon.dev.0 amazon $PROCESSED_DATA_DIR
/home/xgg/anaconda3/envs/py27/bin/python2.7 preprocess_frequency_ratio.py raw_data/amazon/sentiment.test.0 amazon.train.0 label 10000 1 amazon.test.0 amazon $PROCESSED_DATA_DIR
/home/xgg/anaconda3/envs/py27/bin/python2.7 preprocess_train.py raw_data/amazon/sentiment.train.1 amazon.train.1 label 10000 1 amazon.train.1 amazon $PROCESSED_DATA_DIR
/home/xgg/anaconda3/envs/py27/bin/python2.7 preprocess_train.py raw_data/amazon/sentiment.dev.1 amazon.train.1 label 10000 1 amazon.dev.1 amazon $PROCESSED_DATA_DIR
/home/xgg/anaconda3/envs/py27/bin/python2.7 preprocess_frequency_ratio.py raw_data/amazon/sentiment.test.1 amazon.train.1 label 10000 1 amazon.test.1 amazon $PROCESSED_DATA_DIR
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

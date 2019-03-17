set -x
cp configs/cbert_yelp_attention_based.config run.config
PROCESSED_DATA_DIR=processed_data_attention_based
python preprocess_attention_based.py raw_data/yelp/sentiment.train.0 label yelp.train.0 yelp $PROCESSED_DATA_DIR
python preprocess_attention_based.py raw_data/yelp/sentiment.dev.0 label yelp.dev.0 yelp $PROCESSED_DATA_DIR
python preprocess_attention_based.py raw_data/yelp/sentiment.test.0 label yelp.test.0 yelp $PROCESSED_DATA_DIR
python preprocess_attention_based.py raw_data/yelp/sentiment.train.1 label yelp.train.1 yelp $PROCESSED_DATA_DIR
python preprocess_attention_based.py raw_data/yelp/sentiment.dev.1 label yelp.dev.1 yelp $PROCESSED_DATA_DIR
python preprocess_attention_based.py raw_data/yelp/sentiment.test.1 label yelp.test.1 yelp $PROCESSED_DATA_DIR
rm $PROCESSED_DATA_DIR/yelp/train.data.label
rm $PROCESSED_DATA_DIR/yelp/dev.data.label
rm $PROCESSED_DATA_DIR/yelp/test.data.label
cat $PROCESSED_DATA_DIR/yelp/yelp.train.*.data.label >> $PROCESSED_DATA_DIR/yelp/train.data.label
cat $PROCESSED_DATA_DIR/yelp/yelp.dev.*.data.label >> $PROCESSED_DATA_DIR/yelp/dev.data.label
cat $PROCESSED_DATA_DIR/yelp/yelp.test.*.data.label >> $PROCESSED_DATA_DIR/yelp/test.data.label
/home/xgg/anaconda3/envs/py27/bin/python2.7 shuffle.py $PROCESSED_DATA_DIR/yelp/train.data.label
/home/xgg/anaconda3/envs/py27/bin/python2.7 shuffle.py $PROCESSED_DATA_DIR/yelp/dev.data.label
cp $PROCESSED_DATA_DIR/yelp/train.data.label.shuffle $PROCESSED_DATA_DIR/yelp/train.data.label
cp $PROCESSED_DATA_DIR/yelp/dev.data.label.shuffle $PROCESSED_DATA_DIR/yelp/dev.data.label


set -x
cp configs/cbert_amazon_attention_based.config run.config
PROCESSED_DATA_DIR=processed_data_attention_based
python preprocess_attention_based.py raw_data/amazon/sentiment.train.0 label amazon.train.0 amazon $PROCESSED_DATA_DIR
python preprocess_attention_based.py raw_data/amazon/sentiment.dev.0 label amazon.dev.0 amazon $PROCESSED_DATA_DIR
python preprocess_attention_based.py raw_data/amazon/sentiment.test.0 label amazon.test.0 amazon $PROCESSED_DATA_DIR
python preprocess_attention_based.py raw_data/amazon/sentiment.train.1 label amazon.train.1 amazon $PROCESSED_DATA_DIR
python preprocess_attention_based.py raw_data/amazon/sentiment.dev.1 label amazon.dev.1 amazon $PROCESSED_DATA_DIR
python preprocess_attention_based.py raw_data/amazon/sentiment.test.1 label amazon.test.1 amazon $PROCESSED_DATA_DIR
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

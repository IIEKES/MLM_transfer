set -x
cp configs/cbert_yelp_fusion_method.config run.config
PROCESSED_DATA_DIR=processed_data_fusion_method
/home/xgg/anaconda3/envs/py27/bin/python2.7 filter_style_ngrams_modified.py raw_data/yelp/sentiment.train. 2 label yelp.train. yelp $PROCESSED_DATA_DIR
cp drg_tf_idf/yelp/sentiment.train.0.tf_idf.orgin $PROCESSED_DATA_DIR/yelp/yelp.train.0.tf_idf.filter
cp drg_tf_idf/yelp/sentiment.train.1.tf_idf.orgin $PROCESSED_DATA_DIR/yelp/yelp.train.1.tf_idf.filter
/home/xgg/anaconda3/bin/python attn_cls_tf_idf.py $PROCESSED_DATA_DIR
cp $PROCESSED_DATA_DIR/yelp/yelp.train.0.tf_idf.attn.label $PROCESSED_DATA_DIR/yelp/yelp.train.0.tf_idf.label
cp $PROCESSED_DATA_DIR/yelp/yelp.train.1.tf_idf.attn.label $PROCESSED_DATA_DIR/yelp/yelp.train.1.tf_idf.label
/home/xgg/anaconda3/envs/py27/bin/python2.7 preprocess_fusion_method.py raw_data/yelp/sentiment.train.0 yelp.train.0 label 7000 1 yelp.train.0 yelp $PROCESSED_DATA_DIR
/home/xgg/anaconda3/envs/py27/bin/python2.7 preprocess_fusion_method.py raw_data/yelp/sentiment.dev.0 yelp.train.0 label 7000 1 yelp.dev.0 yelp $PROCESSED_DATA_DIR
python preprocess_fusion_method_for_test.py raw_data/yelp/sentiment.test.0 yelp.train.0 label 7000 1 yelp.test.0 yelp $PROCESSED_DATA_DIR
/home/xgg/anaconda3/envs/py27/bin/python2.7 preprocess_fusion_method.py raw_data/yelp/sentiment.train.1 yelp.train.1 label 7000 1 yelp.train.1 yelp $PROCESSED_DATA_DIR
/home/xgg/anaconda3/envs/py27/bin/python2.7 preprocess_fusion_method.py raw_data/yelp/sentiment.dev.1 yelp.train.1 label 7000 1 yelp.dev.1 yelp $PROCESSED_DATA_DIR
python preprocess_fusion_method_for_test.py raw_data/yelp/sentiment.test.1 yelp.train.1 label 7000 1 yelp.test.1 yelp $PROCESSED_DATA_DIR
/home/xgg/anaconda3/bin/python process_unable.py $PROCESSED_DATA_DIR
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


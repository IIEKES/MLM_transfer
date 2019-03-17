set -x
PROJECTPATH=$HOME/pros/MLM_transfer
cp configs/bert_yelp_frequency_ratio.config run.config
PYTHONPATH=$PROJECTPATH $PYTHON_HOME/python fine_tune_bert.py
cp configs/cbert_yelp_frequency_ratio.config run.config
PYTHONPATH=$PROJECTPATH $PYTHON_HOME/python fine_tune_cbert.py
#PYTHONPATH=$PROJECTPATH $PYTHON_HOME/python test_tools/yang_test_tool/cls_wd.py
PYTHONPATH=$PROJECTPATH $PYTHON_HOME/python fine_tune_cbert_w_cls.py

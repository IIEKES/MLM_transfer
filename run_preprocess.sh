set -x
#cp configs/cbert_yelp_attention_based.config run.config
#/home/xgg/anaconda3/bin/python attn_cls_wd.py
#cp configs/cbert_amazon_attention_based.config run.config
#/home/xgg/anaconda3/bin/python attn_cls_wd.py
#bash scripts/attention_based/preprocess_yelp_attention_based.sh
#bash scripts/attention_based/preprocess_amazon_attention_based.sh
bash scripts/frequency_ratio/preprocess_yelp_frequency_ratio.sh
bash scripts/frequency_ratio/preprocess_amazon_frequency_ratio.sh
bash scripts/fusion_method/preprocess_yelp_fusion_method.sh
bash scripts/fusion_method/preprocess_amazon_fusion_method.sh

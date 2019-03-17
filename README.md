# MLM_transfer
Implemetation of MLM_transfer

Environment：
  - python==3.6
  - pytorch==0.4.1
  - theano==1.0.4
  - nltk==3.0.0b2 (included)
  
Procedures：
### Mask stage
- Three methods：
  - attention_based method
  - frequency_ratio method
  - fusion_method method
- Command（Results already included, no need to run it again）
  bash run_preprocess.sh 

### Fill stage
- Two steps：
  - MLM -> fine_tune_cbert.py
  - MLM-SS -> fine_tune_cbert_w_cls.py
  - ~~MLM-PG -> fine_tune_cbert_w_cls_pg.py~~
  
- Commands
  - Corresponds to attention_based mask method
    - bash scripts/attention_based/fine_tune_yelp_attention_based.sh
    - bash scripts/attention_based/fine_tune_amazon_attention_based.sh
   
  - Corresponds to frequency_ratio mask method
    - bash scripts/frequency_ratio/fine_tune_yelp_frequency_ratio.sh
    - bash scripts/frequency_ratio/fine_tune_amazon_frequency_ratio.sh
    
  - Corresponds to fusion_method mask method
    - bash scripts/fusion_method/fine_tune_yelp_fusion_method.sh
    - bash scripts/fusion_method/fine_tune_amazon_fusion_method.sh
    
 Note: The accuracy results produced here are lower than original paper, but the BLEU scores are higher. It is a trade-off between 
 accuracy and BLEU. To achieve the same results from paper, you just need to modify fine_tune_cbert_w_cls.py:
 if lm_loss.item() > 1.5: => if lm_loss.item() > 1.7 or higher # line 153
 
 We also tried to use policy_gradient instead of soft-sampling to back-propagate gradient, and we encourage you to implement it yourself.

With window_size=5, max_depth=15, min_samples_split=10. (*note must adjust window size in SSpred.py for them to match the parameters file)
We end up with 66.98582% accuracy. 
Lower max_depth should reduce overfitting, if the current tree is too complex. 
Increasing min_samples_split should prevent the tree from splitting nodes with few samples, so perhaps better generalization. 

Could also use the entropy loss function rather than the gini. 

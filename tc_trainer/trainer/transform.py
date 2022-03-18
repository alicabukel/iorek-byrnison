
from tensorflow import feature_column  as fc
import numpy as np

from trainer.constants import VOCAB_DICT

def transform(inputs, numeric_cols, nbuckets):
    
    transformed = inputs.copy()

    feature_columns = {
        colname: fc.numeric_column(colname) for colname in numeric_cols
    }
  
    for col, vocab_list in VOCAB_DICT.items():
        feature_columns[col] = fc.indicator_column(
            fc.categorical_column_with_vocabulary_list(col,vocabulary_list=vocab_list)
        )

    buckets = np.linspace(0, 1, nbuckets).tolist()
    b_tenure = fc.bucketized_column(
        feature_columns["tenure"], buckets
    )
    
    b_mc = fc.bucketized_column(
        feature_columns["MonthlyCharges"], buckets
    )
    
    b_tc = fc.bucketized_column(
        feature_columns["TotalCharges"], buckets
    )
   

    ten_mc = fc.crossed_column([b_tenure, b_mc], nbuckets * nbuckets)
    ten_bc = fc.crossed_column([b_tenure, b_tc], nbuckets * nbuckets)
    
    ten_pair = fc.crossed_column([ten_mc, ten_bc], nbuckets ** 4)
    feature_columns["tenure_charges"] = fc.embedding_column(ten_pair, 100)

    return transformed, feature_columns

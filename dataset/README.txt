Read cols_name.npy => Dictionary - name of feature columns, and target columns
cols_name = np.load('cols_name.npy', allow_pickle=True)

Read parquet file
train_df = pd.read_parquet('train_data.parquet')

Metric: spearmanr. Compute score for each Moon, and then get average.
    https://docs.scipy.org/doc/scipy-1.14.0/reference/generated/scipy.stats.spearmanr.html

More info will be followed later.

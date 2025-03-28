# AA2_EDSC
Emory Data Science Club baggage prediction project for American Airlines

Currently, in sasha folder:
* data loading, data cleaning, data combination in data_loader.py >> final_training_data.parquet
  - Lucian Mirja made good VSCode extension for sql querying parquet/csv data
* model architecture in model.py
* training loop in train.py
* overnight_tune.py is long hyperparameter tuning -- 288 combinations
  - implemented early stopping for speed
* expanded_hyperparameter_tune.csv holds all models, with a tuple of validation loss; rank by lowest to find best model
* script.py finds best test model, tests against big dataset of flights, outputs evaluation metrics >> test_set_evaluations.py and some pngs into subdir

Current metrics, evaluated on best model hyperparameters:
Total Test Samples:        24,711

Average KL Divergence:     0.42

Average MSE (Count):       7,622.39

Average RMSE (Count):      87.31

Average MAE (Count):       65.80

Count Error Std Dev:       57.39

Count Error Median:        53.06

Count Error Max:           741.66

Next steps:
* look at evaluation metrics and see what is good, bad, how fix
   - talk to somebody that understands better than me; Ken, Jackson, AA point person (?)
* long-term: theres a lot more we can do with this model in terms of business insights; find new combinations of input features and intuitively correlated output features so that we can give more than just baggage check-in staffing.

Research Papers Referenced:

Entity Embeddings for Categorical Variables
Guo, C., & Berkhahn, F. (2016). Entity Embeddings of Categorical Variables.
arXiv:1604.06737
Demonstrates how learned embeddings can effectively represent high-cardinality categorical variables.

Cyclical Learning Rates
Smith, L. N. (2017). Cyclical Learning Rates for Training Neural Networks.
arXiv:1506.01186
Describes the cyclical learning rate strategy to improve convergence without extensive manual tuning.

Revisiting Deep Learning Models for Tabular Data
Gorishniy, S., Rubachev, I., Khrulkov, V., & Babenko, A. (2021). Revisiting Deep Learning Models for Tabular Data.
arXiv:2106.11959
Highlights how carefully designed feed-forward networks (with batch normalization, dropout, and residual connections) can rival tree-based models on tabular datasets.

Decoupled Weight Decay Regularization (AdamW)
Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization.
arXiv:1711.05101
Introduces AdamW, an optimizer that decouples weight decay from the learning rate, often yielding better generalization than Adam.

Hyperparameter Optimization
Bergstra, J., & Bengio, Y. (2012). Random Search for Hyper-Parameter Optimization.
Journal of Machine Learning Research, 13, 281–305.
Link to PDF
Discusses the effectiveness of random and grid search approaches for hyperparameter tuning, which underpins your hyperparameter search process.

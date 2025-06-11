# Probabilistic Intraday Electricity Price Forecasting Using Generative Machine Learning

This repository provides python codes for the dimensionality reduction approaches to compressing ensemble forecast fields accompanying the paper

>

## Explanation of the code files

- For reproducing the reconstructed forecast fields presented in the paper:

|File name| Explanation |
|-------------|---------------|
|**`lasso.py`**| Python script to implement LASSO for univariate point forecasts. |
|**`lasso_bootstrap.py`**| Python script to implement Step 2 of LASSO bootstrap approach. |
|**`lqc_qr.py`**| Python script to implement Steps 2 of LQC approach. |
|**`lqc_gca.py`**| Python script to implement Steps 3 of LQC approach. |
|**`cgm_models.py`**| Python script of the CGM framework (ES loss and custom loss). |
|**`cgm_epf.py`**| Python script to implement the CGM approach. |
|**`eval_maxindex.py`**| Python script to evaluate using the majority vote strategy. |
|**`eval_trading.py`**| Python script to evaluate using the prediction band-based strategy. |
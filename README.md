<h3 align="center">LSTM - Attention Stock Prediction Model</h3>

----- 

⚠️ **Disclaimer:** This project is for **research and educational purposes only**.  
It is **not a financial advice tool** and should not be used for making real investment or trading decisions.

The main model has infrasturture of to concepts: **LSTM and Multi-head Self Attention mechanism.** With using this concepts shows that is more efficient than using the concepts one by one. An another [Sequential LSTM-Multihead attention model](https://github.com/Amritpal-001/Stock-price-predicition/blob/master/2%20-%20Opening_Price_Prediction_keras.ipynb) was compared with previous model, it turns out second model is more efficient for stock market forecasting prediction purpose. Also, a preprocess technique, [ABBA](https://github.com/nla-group/slearn/blob/master) "Adaptive Brownian bridge-based symbolic Aggregation" was considered for putting into preprocessing process of models itself. However, the results are demonstrating that ABBA isn't a proper technique for model preprocessing. Thats for, a default scaler MinMaxScaler used for both of them. At the end, for hyperparameters, BO (Bayesian Optimization) have been used. All of the implementation of all considerations are been included at seperated python files (Purposes declared on files' titles.). Also, the process of building and selecting the model and processes .ipynb file includes all of the journey of this project. For further information of external resources, check **References** section.

<h4>Install</h4>

<i>This project hasn't any build dynamics. So, all of the repository must be cloned to local environment: </i>

**Installing all components:**
```
pip install torch pytorch_forecasting skopt yfinance numpy pandas matplotlib
```

**For utilizing ABBA (Additional)**:
```
pip install fABBA
```

<h4>Usage</h4>

<i>info: For model behaviour and performance, check plots at Resources/Images directory. All of the current model's loss information and validation comparisons are included. Also, starting with 'S' character titles representing 'single' prediction format.</i>

After the main file ran, selection output will be appeared:

**Output:**
```
Please enter ticker and format for prediction. (AAPL, 1; NVDA, 20;'0 (e.g. single prediction to AAPL, 4 week prediction for NVDA)):
```

**Two models** were provided for prediction: For single day and for a month. Each model can be chosen by referancing their format types, like (ticker, 1: single day prediction/ 20: a month's prediction). E.g.; (AAPL, 1;) is referencing next day's close price of Apple market, (AMZN, 20;) is referencing close prices of next month.

**Input:**
```
AAPL, 1; AAPL, 20; AMZN, 1; AMZN, 20; NVDA, 1; NVDA, 20;
```

Successful formats will be illustrated via plots at Resources/Predictions directory, if that format isn't exist or there aren't any format mistakes. 

<br/>

**Main Purpose & Contributing:**

This side project is mainly focused on models behaviour on forecasting time series and comparing each model's structure with others and optimizing all of them. That's why purposed for technical side, not financial. Contributing to this repo can add unique and diverse techniques for finding the best model for main purpose.

**References:**

Full list of references and external resources is available in
[REFERENCES.md](https://github.com/EralpD/LSTMAttnStockPrediction/blob/main/REFERENCES.md).
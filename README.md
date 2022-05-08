# Appendix
Authors: MCM2022 Group2209354. Real identification could not be published now for `competition fairness`.

This repo is the Appendix of Group2209354 (the last pulling request was published before `DEADLINE`).

In this repo, the file structure is as follows:

```Powershell
root/
|--datasets/ --"datasets"
|	|--Gold.npy --"Gold data after preprocessing"
|	|--Bitcoin.npy --"Bitcoin data after preprocessing"
|	|--BCHAIN-MKPRU.csv
|	|--LBMA-GOLD.csv
|	|--preprocess.py --"to preprocess original .csv data"
|	|--date_map.json --"the map from date to date encode"
|	|--fill_missing_date.m --"to fill the missing data of Gold"
|	|--position_encoding.npy --"the date encode"
|	|--utils/
|		|--visual.py --"visualise the results"
|--dataloader.py --"a tool to load data from datasets/"
|--predict.py --"the predictor of LSTM model"
|--run.py --"the runner of LSTM model"
|--lstm.py --"the LSTM model"
|--dynamic_programming/
|	|--DP.py --"the Heuristic Dynamic Programming to give a final solution"
|	|--utils.py --"the calculator of expectation for DP"
|	|--dataloader.py --"a tool to load data from prediction/"
|	|--prediction/ --"the prediction of t days based on previous n days"
|		|--0.npy
|		|--1.npy
|		|--..
|		|--1937.npy
|		|--gold_allow.npy --"the trading date of gold"
|--comparison/ --"different prediction model to beat with"
	|--LIP.py --"with Lagrange Interpolation Polynomial model"
	|--SVM.py --"with Support Vector Machine model"
	|--Grey.m --"with Grey Prediction model" 
	|--MAE_judger.py --"Mean Average Error calculator"
```

The datasets need to be preprocessed with `preprocess.py` and `fill_missing_date.m`.

The model then can be run with `run.py` to generate the prediction `/dynamic_programming/prediction` as:

```powershell
python3 run.py --type-test --visual-False
```

 Then to run the dynamic programming model with `DP.py` to generate the complete results as:

```powershell
python3 DP.py
```

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

It's not as expected that we finally got an Hornerable Prize. But we've learned a lot from this tiring but happy experience!

The PDF file is uploaded now.

![26021e129cfb491ed9d6bc49684f0e2](https://user-images.githubusercontent.com/79912692/167284576-8d8da925-3a31-4d8b-b2f5-2aff95179f30.png)

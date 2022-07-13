#load the dataset
import pandas as pd
data = pd.read_csv("sample_data.csv")

#Install the package
pip install auto-ts-ensemble
from auto_ts_ensemble_rupakbob import auto_ts_ensemble

#call the package function 1
results = auto_ts_ensemble.neural_analysis(data,freq="H")

###access the model
neural_prophet = results[0]

###access the metrics
metrics = results[1]

###access the predictions
predictions = results[2]

####plot the components
neural_prophet.plot_components(predictions)

#--------------------------------------------
#call the package function 2
results2 = auto_ts_ensemble.ts_analysis(data,n_future=7)
#access the model
ts_model = results2[0]
#access the predictions
predictions2 = results2[1]
#plot the components
ts_model.plot_components(predictions2)

#---------------------------------------------------
#call the package function 3 ensemble
ensemble_predictions = auto_ts_ensemble.ensemble_analysis(predictions,predictions2)
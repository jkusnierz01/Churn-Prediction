Customer churn prediction is one of datasets from kaggle.
I trained 4 models (SVM, RandomForest, XGBoost, MLP, CatBoost). 
I prepared simple web page with Streamlit where you can see what features were important for model in training (SHAP library). Also API is provided with FAST API library.

Under *app* directory you can find *Streamlit* setup.

```fast-api.py``` is file which contains Fast API setup.

All data cleaning and preprocessing was done in jupyer-notebook ```churn-prediction.ipynb```

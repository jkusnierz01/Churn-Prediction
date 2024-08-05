import streamlit as st
import catboost as cb
import pandas as pd

model_path = '../model/catmodel.cbm'

@st.cache_resource
def load_model():
    model = cb.CatBoostClassifier()
    model.load_model(model_path)
    return model

def load_data(file_path):
    df = pd.read_csv(file_path)
    df.reset_index(drop=True,inplace=True)
    return df

def get_model():
    return load_model()
    

if __name__ == "__main__":
    # Load the model into session state
    if 'model' not in st.session_state:
        st.session_state['model'] = get_model()

    pg = st.navigation([
    st.Page("1_MainPage.py", title="Main Page", icon="🔥"),
    st.Page("2_SecondPage.py", title="Second page"),
    st.Page("3_ThirdPage.py", title="Third page"),
    ])

    pg.run()
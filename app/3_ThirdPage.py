import streamlit as st
from streamlit_app import load_data
import shap
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def get_shap_values(model,train_data):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(train_data)

    return explainer, shap_values



X_train = load_data("../X_train.csv")
Y_train = load_data("../Y_train.csv")
X_test = load_data("../X_test.csv")
Y_test = load_data("../Y_test.csv")




model = st.session_state['model']


exp, values = get_shap_values(model,X_train)

shap.summary_plot(values,X_train, plot_type="bar", show=False)
summary_fig, _ = plt.gcf(), plt.gca()
st.pyplot(summary_fig)
plt.close()



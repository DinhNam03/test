import os
import streamlit as st
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import shap


def show():
    BASE_DIR = os.path.dirname(__file__)

    st.title("🔮 Dự đoán viêm tụy cấp")
    st.write("Nhập các chỉ số lâm sàng của bệnh nhân để dự đoán khả năng sống sót:")

    @st.cache_resource
    def load_model():
        return joblib.load('rfc.joblib')

    @st.cache_data
    def load_data():
        return joblib.load('X_train.joblib')
    rfc = load_model()
    X_train = load_data() # type: ignore

    explainer = shap.Explainer(model=rfc, masker=X_train, feature_names=X_train.shape)

    # INPUT FORM 
    col1, col2 = st.columns(2)
    vars = {}

# 'age', 'hypertension', 'wbc', 'rdw', 'bicarbonate', 'creatinine',
#        'alt', 'alp', 'ast', 'inr', 'sepsis', 'aki'
        
    with col1:
        vars['age'] = st.number_input('Age', min_value=20, max_value=93, step=1)
        vars['hypertension'] = st.selectbox('Hypertension', [(0, 'No'), (1, 'Yes')], format_func=lambda v: v[1])[0]
        vars['wbc'] = st.number_input('WBC', min_value=1.7, max_value=12500.0, step=0.1)
        vars['rdw'] = st.number_input('RDW', min_value=12.0, max_value=34.9, step=0.1)
        vars['bicarbonate'] = st.number_input('Bicarbonate', min_value=8.6, max_value=36.01, step=0.1)
        vars['creatinine'] = st.number_input('Creatinine', min_value=0.3, max_value=15.2, step=0.1)
        
    with col2:
        vars['alt'] = st.number_input('ALT', min_value= 6, max_value=9582, step=1)
        vars['alp'] = st.number_input('ALP', min_value=36, max_value=5006, step=1)
        vars['ast'] = st.number_input('AST', min_value=10, max_value=33840, step=1)
        vars['inr'] = st.number_input('INR', min_value=0.9, max_value=15.6, step=0.1)
        vars['sepsis'] = st.selectbox('Sepsis', [(0, 'No'), (1, 'Yes')], format_func=lambda v: v[1])[0]
        vars['aki'] = st.selectbox('AKI', [(0, 'No'), (1, 'Yes')], format_func=lambda v: v[1])[0]
        
        

    st.markdown("---")
    arr = ['Survival', 'Died']

    # PREDICTION 
    if st.button('🔍 Predict'):
        # st.subheader("🧾 Input Summary")
        df_pred = pd.DataFrame([vars])
        # st.write(df_pred.iloc[0])
        pred = rfc.predict(df_pred.iloc[:1])[0]
        pred_prob = rfc.predict_proba(df_pred.iloc[:1])[0]
        st.write(df_pred.iloc[-1:])

        st.subheader("🧠 Prediction Result")
        st.success(f"**Prediction:** {arr[pred]}")
        st.info(f"**Probability:** {pred_prob[pred]:.2f}")

        shap_values = explainer(df_pred.iloc[:1])
        #shap.plots.w
        fig = shap.plots.force(shap_values[0, :, 1], matplotlib=True)
        st.pyplot(fig)
        #fig = shap.plots.waterfall(shap_values[0, :, 1], matplotlib=True)
        #st.pyplot(fig)
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0, :, 1], show=False)
        st.pyplot(fig)
        
       
            

    # st.markdown("---")
    # st.caption("Model: Random Forest Classifier (RFC)")



# age             20.00000
#  hypertension     0.00000
#  wbc              1.70000
#  rdw             12.00000
#  bicarbonate      8.60473
#  creatinine       0.30000
#  alt              6.00000
#  alp             36.00000
#  ast             10.00000
#  inr              0.90000
#  sepsis           0.00000
#  aki              0.00000
#  dtype: float64,
#  age                93.000000
#  hypertension        1.000000
#  wbc             12500.000000
#  rdw                34.900000
#  bicarbonate        36.012871
#  creatinine         15.200000
#  alt              9582.000000
#  alp              5006.000000
#  ast             33840.000000
#  inr                15.600000
#  sepsis              1.000000
#  aki                 1.000000









# import streamlit as st
# import pandas as pd
# import joblib
# import os

# def show():
#     BASE_DIR = os.path.dirname(__file__)
#     @st.cache_resource
#     def load_model():
#         return joblib.load(os.path.join(BASE_DIR, 'rfc1.joblib'))
#     rfc = load_model()
#     st.title('AP Predict')
#     col1, col2 = st.columns(2)
#     vars = {}
#     with col1:
#         vars['bilirubin_total_max'] = st.number_input(label='Bilirubin total', min_value=-23.2, max_value=51.2, step=0.1)
#         vars['rdw_max'] = st.number_input(label='RDW', min_value=11.8, max_value=34.9, step=0.1)
#         vars['NPAR'] = st.number_input(label='NPAR', min_value=1.36, max_value=71.5, step=0.1)
#         vars['NLR'] = st.number_input(label='NLR', min_value=0.04, max_value=270.2, step=0.1)
#         vars['sapsii'] = st.number_input(label='SAPSII', min_value=6, max_value=94, step=1)
#         vars['sofa'] = st.number_input(label='SOFA', min_value=0, max_value=21, step=1)
#     with col2:
#         vars['cci'] = st.number_input(label='CCI', min_value=0, max_value=17, step=1)
#         vars['apsiii'] = st.number_input(label='APSIII', min_value=7, max_value=159, step=1)
#         vars['temperature_mean'] = st.number_input(label='Temperature body', min_value=33.6, max_value=40.1, step=0.1)
#         vars['vasopressin'] = st.selectbox(label='Vasopressin', options=[(0, 'No'), (1, 'Yes')], format_func=lambda v: v[1])[0]
#         vars['crrt'] = st.selectbox(label='CRRT', options=[(0, 'No'), (1, 'Yes')], format_func=lambda v: v[1])[0]
#         vars['has_sepsis'] = st.selectbox(label='Has sepsis', options=[(0, 'No'), (1, 'Yes')], format_func=lambda v: v[1])[0]

#     arr = ['Survival', 'Died']

#     if st.button('Predict'):
#         df_pred = pd.DataFrame([vars])
#         # st.write(df_pred.iloc[0])
#         pred = rfc.predict(df_pred.iloc[:1])[0]
#         pred_prob = rfc.predict_proba(df_pred.iloc[:1])[0]
#         st.write(df_pred.iloc[-1:])

#         st.write(f'Predict: {arr[pred]}, Probability: {pred_prob[pred]:.2f}')






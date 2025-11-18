import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
import pandas as pd
import joblib
import os

def show():
    BASE_DIR = os.path.dirname(__file__)
    st.title("📊 Model Evaluation & ROC Curves")

    @st.cache_resource
    def load_data():
        X = pd.read_csv(os.path.join(BASE_DIR, 'X_test.csv'))
        Y = pd.read_csv(os.path.join(BASE_DIR, 'Y_test.csv'))
        return X, Y

    @st.cache_resource
    def load_models():
        return {
            'Ada Boost': joblib.load(os.path.join(BASE_DIR, 'AdaBoost.joblib')),
            'Extra Trees': joblib.load(os.path.join(BASE_DIR, 'ExtraTrees.joblib')),
            'Gradient Boosting': joblib.load(os.path.join(BASE_DIR, 'GradientBoosting.joblib')),
            'Random Forest': joblib.load(os.path.join(BASE_DIR, 'RandomForest.joblib'))
        }

    X_test, Y_test = load_data()
    models = load_models()

    # ================== SELECT OPTIONS ==================
    mortalities = ['mortality_7d', 'mortality_28d', 'mortality_90d', 'mortality_1y']
    titles = ['7 days', '28 days', '90 days', '1 year']

    st.write("So sánh đường cong ROC của các mô hình học máy qua các mốc thời gian tử vong:")

    selected_models = st.multiselect(
        "Chọn mô hình hiển thị:",
        options=list(models.keys()),
        default=list(models.keys())
    )

    # ================== PLOT ==================
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    for i, mortality in enumerate(mortalities):
        y_test = Y_test[mortality]
        for name in selected_models:
            RocCurveDisplay.from_estimator(models[name], X_test, y_test, ax=axes[i], name=name)
        axes[i].set_title(titles[i])
        axes[i].legend(fontsize=8)
    st.pyplot(fig)

    st.markdown("---")
    st.caption("ROC curves for multiple mortality endpoints using different classifiers.")
    






# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.metrics import RocCurveDisplay
# import joblib
# import os

# def show():
#     BASE_DIR = os.path.dirname(__file__)
#     @st.cache_resource
#     def load_data():
#         X = pd.read_csv(os.path.join(BASE_DIR, 'X_test.csv'))
#         Y = pd.read_csv(os.path.join(BASE_DIR, 'Y_test.csv'))
#         return X, Y
#     @st.cache_resource
#     def load_models():
#         return {
#             'Ada Boost': joblib.load(os.path.join(BASE_DIR, 'AdaBoost.joblib')),
#             'Extra Trees': joblib.load(os.path.join(BASE_DIR, 'ExtraTrees.joblib')),
#             'Gradient Boosting': joblib.load(os.path.join(BASE_DIR, 'GradientBoosting.joblib')),
#             'Random Forest': joblib.load(os.path.join(BASE_DIR, 'RandomForest.joblib'))
#         }
#     X_test, Y_test = load_data()
#     models = load_models()

#     st.title("📊 Model Evaluation")
#     mortalities = ['mortality_7d', 'mortality_28d', 'mortality_90d', 'mortality_1y']
#     titles = ['7 days', '28 days', '90 days', '1 year']
#     fig, axes = plt.subplots(1, 4, figsize=(20, 6))
#     for i, m in enumerate(mortalities):
#         y_test = Y_test[m]
#         for name, model in models.items():
#             RocCurveDisplay.from_estimator(model, X_test, y_test, ax=axes[i], name=name)
#         axes[i].set_title(titles[i])
#     st.pyplot(fig)


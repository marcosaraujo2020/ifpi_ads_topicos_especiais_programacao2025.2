# Iris_interativo.py
import streamlit as st
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import plotly.graph_objs as go

st.set_page_config(page_title="Classificação Iris Interativa", layout="wide")
st.title("🌸 Classificação Iris: versicolor (1) vs virginica (0)")

# 1) Carregar base Iris e tornar binária
iris = load_iris()
X_full = iris.data
y_full = iris.target
mask = (y_full != 0)
X = X_full[mask]
y = (y_full[mask] == 1).astype(int)  # versicolor=1, virginica=0
cols = [2, 3]  # petal length, petal width
X2 = X[:, cols]
feature_names = [iris.feature_names[i] for i in cols]

st.write("**Features usadas:**", feature_names)

# 2) Separar treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X2, y, test_size=0.30, random_state=42, stratify=y
)
st.write(f"Tamanhos -> treino: {X_train.shape[0]} | teste: {X_test.shape[0]}")

# 3) Pipeline
modelo = make_pipeline(StandardScaler(), LogisticRegression(random_state=42))
modelo.fit(X_train, y_train)

# 4) Avaliação
y_pred = modelo.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, pos_label=1)
rec = recall_score(y_test, y_pred, pos_label=1)
f1 = f1_score(y_test, y_pred, pos_label=1)

st.subheader("Desempenho no TESTE")
st.write(f"Acurácia : {acc:.3f}")
st.write(f"Precisão : {prec:.3f}")
st.write(f"Recall   : {rec:.3f}")
st.write(f"F1       : {f1:.3f}")
st.write("**Matriz de Confusão**")
st.write(cm)

# Mostrar matriz de confusão como tabela
cm_df = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]) * 100
cm_df = np.round(cm_df, 2)
st.write("**Matriz de Confusão (%)**")
st.dataframe(cm_df, use_container_width=True)



# 5) Função para plot interativo
def plot_fronteira_interativa(modelo, X_all, X_set, y_set, title):
    # grade
    h = 0.02
    x_min, x_max = X_all[:,0].min() - 0.5, X_all[:,0].max() + 0.5
    y_min, y_max = X_all[:,1].min() - 0.5, X_all[:,1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = modelo.predict(grid).reshape(xx.shape)

    # criar figura plotly
    fig = go.Figure()

    # background das classes
    fig.add_trace(go.Contour(
        z=Z,
        x=np.arange(x_min, x_max, h),
        y=np.arange(y_min, y_max, h),
        showscale=False,
        colorscale=[[0, 'lightblue'], [1, 'lightpink']],
        opacity=0.4,
        contours=dict(showlines=False)
    ))

    # pontos de treino/teste
    fig.add_trace(go.Scatter(
        x=X_set[:,0], y=X_set[:,1],
        mode='markers',
        marker=dict(color=y_set, colorscale=['blue','red'], line=dict(width=1, color='black'), size=10),
        text=[f"Classe: {'versicolor' if label==1 else 'virginica'}" for label in y_set],
        hoverinfo='text'
    ))

    fig.update_layout(title=title,
                      xaxis_title=feature_names[0],
                      yaxis_title=feature_names[1],
                      width=700, height=500)
    return fig

st.subheader("Fronteira de decisão (interativa)")
fig_train = plot_fronteira_interativa(modelo, X2, X_train, y_train, "Treino")
st.plotly_chart(fig_train)
fig_test = plot_fronteira_interativa(modelo, X2, X_test, y_test, "Teste")
st.plotly_chart(fig_test)

# 6) Predições individuais
st.subheader("Predições individuais")
petal_length = st.number_input("Petal length (cm)", min_value=float(X2[:,0].min()), max_value=float(X2[:,0].max()), value=4.5)
petal_width  = st.number_input("Petal width (cm)", min_value=float(X2[:,1].min()), max_value=float(X2[:,1].max()), value=1.3)

sample = np.array([[petal_length, petal_width]])
probas = modelo.predict_proba(sample)[0]
pred = modelo.predict(sample)[0]

st.write(f"Predição: **{'versicolor (1)' if pred==1 else 'virginica (0)'}**")
st.write(f"Probabilidades -> virginica (0): {probas[0]:.3f}, versicolor (1): {probas[1]:.3f}")

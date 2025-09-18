# ===============================================================
# App Streamlit - Covertype binário (Spruce/Fir=1 vs outros=0)
# Modelo: Regressão Logística
# Saídas:
#   - métricas e matriz de confusão
#   - gráficos de fronteira de decisão (treino/teste)
#   - predições individuais
# ===============================================================

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd


st.title("🌲 Classificação de Árvores - Covertype (Spruce/Fir vs Outros)")

# 1) Carregar base
data = fetch_covtype()
X_full = data.data
y_full = data.target  # valores de 1 a 7

# Transformar em binário: 1 = Spruce/Fir, 0 = outros
y = (y_full == 1).astype(int)

# Usar só 2 features para visualização
cols = [0, 2]  # Elevation (0) e Horizontal_Distance_To_Hydrology (2)
X2 = X_full[:, cols]
feature_names = [data.feature_names[i] for i in cols]

# 2) Separar em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X2, y, test_size=0.30, random_state=42, stratify=y
)

# 3) Pipeline: padronizar + Regressão Logística
modelo = make_pipeline(
    StandardScaler(),
    LogisticRegression(random_state=42)
)
modelo.fit(X_train, y_train)

# 4) Avaliar
y_pred = modelo.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, pos_label=1)
rec  = recall_score(y_test, y_pred, pos_label=1)
f1   = f1_score(y_test, y_pred, pos_label=1)

st.subheader("📊 Desempenho no TESTE")
st.write(f"**Acurácia:** {acc:.3f}")
st.write(f"**Precisão:** {prec:.3f}")
st.write(f"**Recall:** {rec:.3f}")
st.write(f"**F1-score:** {f1:.3f}")

st.write("**Matriz de Confusão** (linhas = Real, colunas = Previsto):")
st.text(f"           Prev 0   Prev 1\n"
        f"Real 0  |   {cm[0,0]:>5}   {cm[0,1]:>5}   <- Outros\n"
        f"Real 1  |   {cm[1,0]:>5}   {cm[1,1]:>5}   <- Spruce/Fir\n")


cm_df = pd.DataFrame(
    cm,
    index=["Real 0 (Outros)", "Real 1 (Spruce/Fir)"],
    columns=["Prev 0 (Outros)", "Prev 1 (Spruce/Fir)"]
)
st.table(cm_df)


# 5) Função para plote da fronteira
def plot_fronteira(modelo, X_all, X_set, y_set, title):
    h = 10
    x_min, x_max = X_all[:,0].min() - 50, X_all[:,0].max() + 50
    y_min, y_max = X_all[:,1].min() - 50, X_all[:,1].max() + 50
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = modelo.predict(grid).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6, 4.5))  # cria figura e eixos
    ax.contourf(xx, yy, Z, alpha=0.2)
    ax.scatter(X_set[:,0], X_set[:,1], c=y_set, edgecolors="k", s=20)
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title(title)
    fig.tight_layout()

    st.pyplot(fig)  # passa a figura explicitamente


# 6) Mostrar gráficos
st.subheader("📈 Fronteiras de decisão")
st.write("Treino:")
plot_fronteira(modelo, X2, X_train, y_train, "Fronteira – Treino")

st.write("Teste:")
plot_fronteira(modelo, X2, X_test, y_test, "Fronteira – Teste")

# 7) Predições individuais
st.subheader("🔮 Predições Individuais")

elevation = st.number_input(
    "Elevation",
    min_value=float(X2[:,0].min()),
    max_value=float(X2[:,0].max()),
    value=float(X2[:,0].mean())
)
dist_hydro = st.number_input(
    "Horizontal Distance to Hydrology",
    min_value=float(X2[:,1].min()),
    max_value=float(X2[:,1].max()),
    value=float(X2[:,1].mean())
)

if st.button("Classificar"):
    sample = np.array([[elevation, dist_hydro]])
    probas = modelo.predict_proba(sample)[0]
    pred   = modelo.predict(sample)[0]

    st.write(f"**Entrada:** Elevation={elevation}, Dist_Hydro={dist_hydro}")
    st.write(f"**Previsão:** {'🌲 Spruce/Fir (1)' if pred==1 else 'Outros (0)'}")
    st.write(f"P(Outros=0) = {probas[0]:.3f} | P(Spruce/Fir=1) = {probas[1]:.3f}")

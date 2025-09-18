# pages/Adult.py
# ===============================================================
# Adult binário (>50K=1 vs <=50K=0) com 2 features
# Modelo: Regressão Logística
# Saídas:
#   - métricas e matriz de confusão no teste
#   - figura 1: fronteira de decisão + pontos de TREINO
#   - figura 2: fronteira de decisão + pontos de TESTE
# ===============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

st.title("💼 Classificação de Renda - Adult (>50K vs ≤50K)")

# 1) Carregar a base Adult do OpenML
@st.cache_data
def carregar_dados():
    adult = fetch_openml(name='adult', version=2, as_frame=True)
    df = adult.frame
    df['target'] = (df['class'] == '>50K').astype(int)
    return df

df = carregar_dados()

# Selecionar duas features numéricas
cols = ['age', 'hours-per-week']
X2 = df[cols].values
y = df['target'].values
feature_names = cols

st.write("Problema binário: renda >50K (1) vs ≤50K (0)")
st.write("Features usadas:", feature_names)

# 2) Separar em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X2, y, test_size=0.30, random_state=42, stratify=y
)

st.write(f"Tamanhos -> treino: {X_train.shape[0]} | teste: {X_test.shape[0]}")

# 3) Pipeline: padronizar + Regressão Logística
modelo = make_pipeline(
    StandardScaler(),
    LogisticRegression(random_state=42)
)

# 4) Treinar
modelo.fit(X_train, y_train)

# 5) Avaliar
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
st.write(f"**F1 Score:** {f1:.3f}")

st.write("### Matriz de Confusão")
st.write(pd.DataFrame(
    cm,
    index=["Real ≤50K", "Real >50K"],
    columns=["Prev ≤50K", "Prev >50K"]
))

# 6) Função para plote da fronteira de decisão
def plot_fronteira(modelo, X_all, X_set, y_set, title):
    h = 1
    x_min, x_max = X_all[:,0].min() - 5, X_all[:,0].max() + 5
    y_min, y_max = X_all[:,1].min() - 5, X_all[:,1].max() + 5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = modelo.predict(grid).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6,4.5))
    ax.contourf(xx, yy, Z, alpha=0.2)
    scatter = ax.scatter(X_set[:,0], X_set[:,1], c=y_set, edgecolors="k", s=20)
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title(title)
    return fig

st.subheader("🌐 Fronteiras de Decisão")
st.pyplot(plot_fronteira(modelo, X2, X_train, y_train, "TREINO"))
st.pyplot(plot_fronteira(modelo, X2, X_test, y_test, "TESTE"))

# 7) Predições individuais
st.subheader("🔮 Predições Individuais")
st.write("Forneça valores de entrada para prever:")

age = st.slider("Idade", int(X2[:,0].min()), int(X2[:,0].max()), int(X2[:,0].mean()))
hours = st.slider("Horas por semana", int(X2[:,1].min()), int(X2[:,1].max()), int(X2[:,1].mean()))

sample = np.array([[age, hours]])
proba = modelo.predict_proba(sample)[0]
pred = modelo.predict(sample)[0]

st.write(f"**Entrada:** [Idade={age}, Horas/semana={hours}]")
st.write(f"**Predição:** {'>50K (1)' if pred==1 else '≤50K (0)'}")
st.write(f"P(≤50K) = {proba[0]:.3f} | P(>50K) = {proba[1]:.3f}")

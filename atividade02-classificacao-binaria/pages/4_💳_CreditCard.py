# pages/CreditCard.py
# ===============================================================
# Credit Card Fraud (binário: fraude=1 vs não-fraude=0)
# Modelo: Regressão Logística com 2 features (Amount, Time)
# Saídas:
#   - métricas e matriz de confusão no teste
#   - figura 1: fronteira de decisão + pontos de TREINO
#   - figura 2: fronteira de decisão + pontos de TESTE
# ===============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

st.title("💳 Detecção de Fraude em Cartão de Crédito")

# 1) Carregar dataset
@st.cache_data
def carregar_dados():
    df = pd.read_csv("notebooks/dataset_creditcard/creditcard.csv")  # precisa estar no mesmo diretório
    return df

df = carregar_dados()

# 2) Selecionar features interpretáveis
cols = ["Amount", "Time"]
X2 = df[cols].values
y = df["Class"].values  # 1 = fraude, 0 = não-fraude
feature_names = cols

st.write("Problema binário: fraude (1) vs não-fraude (0)")
st.write("Features usadas:", feature_names)

# 3) Separar em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X2, y, test_size=0.30, random_state=42, stratify=y
)
st.write(f"Tamanhos -> treino: {X_train.shape[0]} | teste: {X_test.shape[0]}")

# 4) Pipeline: padronizar + Regressão Logística
modelo = make_pipeline(
    StandardScaler(),
    LogisticRegression(random_state=42, class_weight="balanced")
)

# 5) Treinar
modelo.fit(X_train, y_train)

# 6) Avaliar
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
    index=["Real não-fraude", "Real fraude"],
    columns=["Prev não-fraude", "Prev fraude"]
))

# 7) Função para plote da fronteira de decisão
def plot_fronteira(modelo, X_all, X_set, y_set, title):
    h = 1000
    x_min, x_max = X_all[:,0].min() - 500, X_all[:,0].max() + 500
    y_min, y_max = X_all[:,1].min() - 5000, X_all[:,1].max() + 5000
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = modelo.predict(grid).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6,4.5))
    ax.contourf(xx, yy, Z, alpha=0.2)
    ax.scatter(X_set[:,0], X_set[:,1], c=y_set, edgecolors="k", s=20)
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title(title)
    return fig

st.subheader("🌐 Fronteiras de Decisão")
st.pyplot(plot_fronteira(modelo, X2, X_train, y_train, "TREINO"))
st.pyplot(plot_fronteira(modelo, X2, X_test, y_test, "TESTE"))

# 8) Predições individuais
st.subheader("🔮 Predições Individuais")
st.write("Forneça valores de entrada para prever:")

amount = st.number_input("💵 Valor da transação (Amount)", float(X2[:,0].min()), float(X2[:,0].max()), float(X2[:,0].mean()))
time   = st.number_input("⏰ Tempo da transação (Time)", float(X2[:,1].min()), float(X2[:,1].max()), float(X2[:,1].mean()))

sample = np.array([[amount, time]])
proba = modelo.predict_proba(sample)[0]
pred = modelo.predict(sample)[0]

st.write(f"**Entrada:** [Amount={amount:.2f}, Time={time:.2f}]")
st.write(f"**Predição:** {'fraude (1)' if pred==1 else 'não-fraude (0)'}")
st.write(f"P(não-fraude) = {proba[0]:.3f} | P(fraude) = {proba[1]:.3f}")

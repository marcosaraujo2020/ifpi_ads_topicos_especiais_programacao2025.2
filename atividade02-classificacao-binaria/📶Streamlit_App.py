import streamlit as st


st.set_page_config(page_title="Classificação Binária", layout="centered")

st.title(" 📘 Tópicos Especiais em Programação - IFPI 2025.2")

#Deixar header e image lado a lado
st.divider()

st.header("🔍 Classificação Binária com Streamlit")

st.write("""
Este projeto permite explorar modelos de classificação binária em diferentes bases de dados:
- **Iris**: versicolor vs virginica
- **Covertype**: Spruce/Fir vs outros
- **Adult**: renda >50K vs ≤50K
- **Credit Card Fraud**: fraude vs não-fraude

Use o menu lateral para navegar entre os experimentos.
""")

#Fazer uma divisao 
st.divider()


st.code("""
print("Hello, Streamlit!")
def add(a, b):
    return a + b
result = add(2, 3)
print(f"Result: {result}")
""", language="python")


st.header("📂 Estrutura do Projeto")
st.button("Clique aqui para ver a estrutura do projeto")
st.audio("https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3")
st.image("https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png", width=200)
st.video("https://www.youtube.com/watch?v=JwSS70SZdyM")
st.spinner()
st.balloons()
st.progress(70)
st.success("Projeto Carregado com Sucesso!")
st.error("Erro ao carregar o projeto!")
st.warning("Aviso: Dados não balanceados!")
st.info("Dica: Use o menu lateral para navegar entre os experimentos.")
st.markdown("**Markdown** _suporta_ [links](https://streamlit.io) e outros elementos.")

number = st.slider("Pick a number", 0, 100)
pets = ["Dog", "Cat", "Bird"]

pet = st.radio("Pick a pet", pets)

date = st.date_input("Pick a date")

color = st.color_picker("Pick a color") 

st.caption("Projeto desenvolvido por Marcos Araújo - IFPI 2025.2")

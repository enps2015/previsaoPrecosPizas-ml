import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv('pizzas.csv')

modelo = LinearRegression()
x = df[["diametro"]]
y = df[["preco"]]

modelo.fit(x, y)

st.title("Prevendo o Valor de uma Pizza")
st.divider()

diametro = st.number_input("Informe o diâmetro da Pizza", min_value=0.0, max_value=100.0, step=0.1, key="diametro")

if diametro:
    preco_previsto = modelo.predict([[diametro]])[0][0]
    st.write(
        f"O valor previsto para a pizza de {diametro:.2f} cm é R$ {preco_previsto:.2f}"
    )
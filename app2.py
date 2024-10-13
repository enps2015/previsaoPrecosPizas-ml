import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt

# Configurações da página
st.set_page_config(page_title="Projeto de Previsão de Preço de Pizzas", layout="wide")

# Paleta de cores personalizada
primary_color = "#FF4B4B"
secondary_color = "#4B4BFF"
bg_color = "#F5F5F5"
text_color = "#333333"
dark_bg_color = "#1F1F1F"  # Fundo escuro
slider_color = "#FFA500"  # Novo tom para o slider

# Função para estilizar a página
def apply_custom_styles():
    st.markdown(f"""
    <style>
        body {{
            background-color: {bg_color};
            color: {text_color};
        }}
        .stButton>button {{
            background-color: {primary_color};
            color: white;
        }}
        .stSlider>div {{
            background-color: {slider_color};
        }}
        .stSlider>div>div>div[data-baseweb="thumb"] {{
            background-color: {text_color} !important;
        }}
        .stSlider>div>div>div[data-baseweb="thumb"]::before {{
            color: {text_color} !important;
        }}
        h1 {{
            color: {primary_color};
        }}
        .css-145kmo2 {{
            padding: 20px;
        }}
    </style>
    """, unsafe_allow_html=True)

apply_custom_styles()

# Leitura dos dados
df = pd.read_csv('dataset/pizzas.csv')

# Separando as variáveis
x = df[["diametro"]]
y = df[["preco"]]

# Criando e treinando o modelo
modelo = LinearRegression()
modelo.fit(x, y)

# Barra lateral
st.sidebar.image("img/pizzaria_logo.png", 
                 caption="Pizzaria do Chef", 
                 use_column_width=True)
st.sidebar.header("Sabores de Pizza 🍕")
sabores = ["Margherita", "Pepperoni", "Calabresa", "Frango c/ Catupiry"]
st.sidebar.write("\n".join(sabores))

# Slider para selecionar o número de fatias
fatias = st.sidebar.slider("Escolha o número de fatias 🍕", min_value=4, max_value=12, step=2, value=8)

# Slider para selecionar o diâmetro da pizza
diametro = st.sidebar.slider("Escolha o diâmetro da pizza (cm)", min_value=20, max_value=200, step=1, value=30)

# Mostrando a imagem da pizza cortada conforme o número de fatias
st.sidebar.image(f"img/{fatias}-fatias.jpg", 
                 caption=f"Pizza com {fatias} fatias", use_column_width=True)

# Prevendo o preço da pizza com base no diâmetro
preco_previsto = modelo.predict([[diametro]])[0][0]
st.sidebar.success(f"O valor estimado para uma pizza de {diametro} cm é R$ {preco_previsto:.2f}")

# Título
st.title("Previsão de Preço de Pizzas 🍕")

# Subtítulo
st.write("Este projeto foi desenvolvido como parte do estudo inicial de machine learning, usando regressão linear para prever os preços de pizzas com base em seu diâmetro.")

# Dividindo a área principal em duas colunas
col1, col2 = st.columns(2)

# Coluna 1: Gráfico de correlação ajustado
with col1:
    st.subheader("Relação Diâmetro x Preço")
    fig, ax = plt.subplots(figsize=(6, 4))  # Tamanho do gráfico ainda menor
    sns.scatterplot(data=df, x='diametro', y='preco', ax=ax, color=primary_color, s=100)
    ax.set_facecolor(dark_bg_color)  # Fundo do gráfico escuro
    ax.set_title("Correlação Diâmetro vs Preço", fontsize=16, color=text_color)
    ax.set_xlabel("Diâmetro (cm)", fontsize=14, color=text_color)
    ax.set_ylabel("Preço (R$)", fontsize=14, color=text_color)
    plt.grid(True, color=bg_color)
    st.pyplot(fig)

# Coluna 2: Imagem da pizza e preço previsto ajustados
with col2:
    # Parte superior: Imagem da pizza com tamanho ajustado
    st.image(f"img/{fatias}-fatias.jpg", 
             caption=f"Pizza com {fatias} fatias", 
             use_column_width=False, width=300)  # Tamanho ajustado da pizza

    # Parte inferior: Caixa de mensagem com o valor previsto e histórico
    st.write("### Valor da Pizza")
    st.success(f"O valor estimado para uma pizza de {diametro} cm é R$ {preco_previsto:.2f}")

    # Histórico de previsões
    if 'previsoes' not in st.session_state:
        st.session_state.previsoes = []
    
    st.session_state.previsoes.append((diametro, preco_previsto))
    
    st.write("### Histórico de Preços Estimados:")
    with st.expander("Ver histórico"):
        for diam, preco in st.session_state.previsoes:
            st.write(f"- Pizza de {diam:.2f} cm: R$ {preco:.2f}")

# Rodapé com links para redes sociais
st.markdown("""
<hr>
<div style='text-align:center;'>
    <a href='https://www.linkedin.com/in/eric-np-santos/' target='_blank' style='text-decoration: none; color: #0077B5; font-size: 24px; margin: 0 15px;'>
        🔗 LinkedIn
    </a>
    <a href='https://github.com/enps2015' target='_blank' style='text-decoration: none; color: #333; font-size: 24px; margin: 0 15px;'>
        💻 GitHub
    </a>
</div>
<p style='text-align:center;'>
&copy; 2024 - Desenvolvido por Eric Pimentel | Obrigado por utilizar o aplicativo de previsão de preços de pizzas!
</p>
""", unsafe_allow_html=True)

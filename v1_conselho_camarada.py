from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from PIL import Image

# Definindo as configurações da página
st.set_page_config(
    page_title="Resumo Camarada", layout="wide", initial_sidebar_state="expanded"
)


# Função para baixar e salvar as notícias atuais
def baixar_noticias():
    today = date.today().strftime("%d%m%Y")
    base_url = (
        "https://www.infomoney.com.br/mercados/ibovespa-hoje-bolsa-de-valores-ao-vivo-"
    )
    url = f"{base_url}{today}/"
    response = requests.get(url)

    if response.status_code == 200:
        content = response.content
        soup = BeautifulSoup(content, "html.parser", from_encoding="utf-8")

        # Encontrando todas as notícias e seus respectivos horários
        news_titles = soup.find_all("h2", class_="live-title")
        news_contents = soup.find_all("div", class_="live-content")
        news_times = soup.find_all("span", class_="material-icons live-published-icon")

        news_text = ""
        # Iterando sobre as notícias, horários e conteúdos e adicionando ao texto
        for title, content, time in zip(
            news_titles[:5], news_contents[:5], news_times[:5]
        ):
            news_text += f"{time.next_sibling.strip()}: {title.get_text(strip=True)}\n\n{content.get_text(strip=True)}\n\n"

        return news_text
    else:
        st.write(
            "Fim de Semana, mercado fechado, aproveite para análisar as ações na aba ao lado!"
        )
        st.write("Caso queria um resumo das última semana, selecione Últimos dias!")
        return None


# Função para baixar e salvar noticias dos ultimos dias
def ultimos_dias(periodo):
    def baixar_noticias_por_data(data):
        base_url = "https://www.infomoney.com.br/mercados/ibovespa-hoje-bolsa-de-valores-ao-vivo-"
        url = f"{base_url}{data}/"
        response = requests.get(url)

        if response.status_code == 200:
            content = response.content
            soup = BeautifulSoup(content, "html.parser", from_encoding="utf-8")

            # Encontrando todas as notícias e seus respectivos horários
            news_titles = soup.find_all("h2", class_="live-title")
            news_contents = soup.find_all("div", class_="live-content")
            news_times = soup.find_all(
                "span", class_="material-icons live-published-icon"
            )

            news_text = ""
            for title, content, time in zip(news_titles, news_contents, news_times):
                news_text += f"{time.next_sibling.strip()}: {title.get_text(strip=True)}\n\n{content.get_text(strip=True)}\n\n"

            return news_text
        else:
            return None

    noticias = ""
    if periodo == "ultimos_7_dias":
        dias = range(0, 3)
    elif periodo == "semana_anterior":
        dias = range(7, 14)
    else:
        return "Período inválido. Escolha 'ultimos_7_dias' ou 'semana_anterior'."

    for i in dias:
        dia = date.today() - timedelta(days=i)
        data_formatada = dia.strftime("%d%m%Y")
        noticias_dia = baixar_noticias_por_data(data_formatada)
        if noticias_dia:
            noticias_dia = noticias_dia[:6000]
            noticias += f"Notícias de {dia.strftime('%d/%m/%Y')}:\n\n{noticias_dia}\n\n"

    if noticias:
        return noticias
    else:
        return "Nenhuma notícia disponível no período solicitado."


st.sidebar.markdown(
    "<p style='font-size:20px;text-align: center'><strong>Seja bem-vindo!</strong></p>",
    unsafe_allow_html=True,
)

# Carregar imagem do logo para exibir na barra lateral
logo_image = Image.open("logo.png")
st.sidebar.image(logo_image, use_column_width=True)

st.sidebar.markdown(
    "<p style='font-size:17px;text-align: center;'>O seu assistente financeiro que irá resumir as principais notícias do mercado! </p>",
    unsafe_allow_html=True,
)

# Adicionando opções na barra lateral
mode = st.sidebar.radio("📰 Notícias:", options=["Hoje", "Últimos dias"], index=0)


# Função para determinar qual função usar para baixar notícias
def selecionar_funcao_baixar_noticias():
    if mode == "Hoje":
        return baixar_noticias()
    elif mode == "Últimos dias":
        return ultimos_dias("ultimos_7_dias")


# Chave de API do OpenAI
load_dotenv()


# Função para gerar respostas do chatbot
def gerar_resposta(input_text):
    news_text = selecionar_funcao_baixar_noticias()
    if news_text:
        embeddings = OpenAIEmbeddings()
        # Criar um Document para o texto de notícias
        document = Document(page_content=news_text)
        documents = [document]
        db = FAISS.from_documents(documents, embeddings)

        # Iniciar ChatOpenAI
        llm = ChatOpenAI(temperature=0.45, model="gpt-4o")

        template = """
                    Você é um investidor/jornalista especializado em escrever artigos sobre o mercado de investimentos. Seu foco abrange ações, fundos imobiliários, criptomoedas, fundos de investimento, Tesouro Direto, debêntures, ETFs (Exchange Traded Funds), imóveis e fundos imobiliários (FIIs), CDI, CDB, poupança, taxa de juros, SELIC, entre outros investimentos. Seu papel é responder perguntas sobre o mercado de forma clara, detalhada e baseada em dados e notícias atuais.
                    Público-Alvo:Suas respostas devem ser acessíveis tanto para investidores iniciantes quanto para experientes.

                    Instruções
                    Formato das Respostas:

                    Perguntas Diretas: Forneça respostas curtas e objetivas, explicando brevemente o motivo da alta ou baixa de uma ação ou cotação de moedas, como o dólar.
                    Desempenho Diário: Resuma o desempenho do mercado e da bolsa de valores em até 7 linhas, oferecendo detalhes suficientes sem se estender demais.
                    Estrutura da Resposta:

                    Use dados e notícias recentes para embasar suas respostas.
                    Para perguntas mais amplas, como o estado geral do mercado, forneça uma visão detalhada mas concisa.
                    Exemplos de Resposta:

                    Pergunta: "Qual foi a ação com o melhor desempenho hoje?"
                    Resposta: "A ação X teve o melhor desempenho hoje devido a [motivo], resultando em uma alta de [percentual]."
                    Aqui está perguntas que pessoas vão fazer para você.
                    {message}

                  E aqui tenho as principais noticias do dia sobre o mercado de investimentos, organizadas por hora.
                    {resume}

                    """

        prompt_template = PromptTemplate(
            input_variables=["message", "resume"], template=template
        )
        chain = LLMChain(llm=llm, prompt=prompt_template)

        similar_response = db.similarity_search(input_text, k=10)
        response = chain.run(message=input_text, resume=similar_response)
        return response
    else:
        return "Não há notícias disponíveis no momento."


# Criando abas
tab1, tab2 = st.tabs(["📰 Notícias do Mercado", "📊 Análise de Dados"])

# Lidando com a interação do usuário com o chatbot
chat_input = st.chat_input("Digite aqui!")

with tab1:
    # Inicializando o histórico de mensagens
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": "Qual sua dúvida sobre o mercado financeiro?",
            }
        ]

    # Exibindo as mensagens do histórico
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Lidando com a interação do usuário com o chatbot
    if chat_input:
        st.session_state.messages.append({"role": "user", "content": chat_input})
        st.chat_message("user").write(chat_input)

        response = gerar_resposta(chat_input)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)

with tab2:
    col1, col2 = st.columns(2)
    df = pd.read_csv(
        r"C:\Users\sensix\Desktop\PESSOAL\PESSOAL\PORTIFOLIO DATA SCIENCE\conselho_camarada\ativos.csv"
    )

    acoes = df["Valor"].unique().tolist()

    # Função para baixar dados usando yfinance
    @st.cache_data
    def baixar_dados(acoes, data_inicial, data_final):
        # Ajustar o formato das datas
        start_date = data_inicial.strftime("%Y-%m-%d")
        end_date = data_final.strftime("%Y-%m-%d")

        # Lista para armazenar os dados das ações
        stock_data = []

        for acao in acoes:
            simbolo_ativo = acao
            if not simbolo_ativo.endswith(".SA"):
                simbolo_ativo += ".SA"

            try:
                # Baixar os dados do ativo usando yf.download
                stock_history = yf.download(
                    simbolo_ativo, start=start_date, end=end_date, interval="1d"
                )

                # Adicionar uma coluna 'Nome' ao DataFrame dos dados do ativo e definir o valor como o nome da empresa
                nome_empresa = df[df["Valor"] == acao]["Empresa"].values[0]
                stock_history["Nome"] = nome_empresa
                stock_history["Ativo"] = acao
                stock_data.append(stock_history)
            except Exception as e:
                print(f"Erro ao baixar dados para o símbolo {simbolo_ativo}: {e}")

        # Concatenar os dados das ações em um único DataFrame
        if stock_data:
            all_stock_data = pd.concat(stock_data)
            all_stock_data["Date"] = pd.to_datetime(all_stock_data.index)

            # Salvar os dados em um arquivo CSV
            all_stock_data.to_csv("dados_acoes_brasileiras.csv")
            return all_stock_data
        else:
            return pd.DataFrame()

    # Criando um st.multiselect
    selecao = col1.multiselect("Selecione as ações 📈:", options=acoes)

    # Definindo a data inicial e final
    data_inicial = date(2018, 1, 1)
    data_final = date.today()

    # Criando um filtro de data
    data_selecionada = col2.date_input(
        "Selecione o período 📅:",
        value=(data_inicial, data_final),
        min_value=data_inicial,
        max_value=data_final,
    )

    # Garantir que as datas selecionadas são um intervalo
    if isinstance(data_selecionada, tuple) and len(data_selecionada) == 2:
        data_inicial, data_final = data_selecionada
    else:
        data_inicial = data_selecionada
        data_final = data_selecionada

    # Baixar os dados automaticamente ao selecionar as ações e o período
    if selecao:
        dados = baixar_dados(selecao, data_inicial, data_final)

        col1, col2 = st.columns(2)
        with col1:
            # Criar um gráfico de linha com a evolução do preço das ações por data
            fig = px.line(
                dados,
                x="Date",
                y="Close",
                color="Nome",
                title="Evolução do Preço das Ações",
                labels={"Date": "Data", "Close": " R$ Preço", "Nome": "Ação"},
            )
            st.plotly_chart(fig)
            # Agregando os dados para criar o gráfico de barras
            volume_agg = dados.groupby("Nome")["Volume"].sum().reset_index()
            fig_volume = px.bar(
                volume_agg,
                x="Volume",
                y="Nome",
                color="Nome",
                orientation="h",
                title="Volume de Ações Negociadas",
                labels={"Volume": "Volume de Negociações", "Nome": "Empresa"},
            )
            st.plotly_chart(fig_volume)

        with col2:
            # Criar o gráfico de candlestick
            # Calcular as médias móveis
            dados["MA20"] = dados["Close"].rolling(window=20).mean()
            dados["MA50"] = dados["Close"].rolling(window=50).mean()

            fig = go.Figure(
                data=[
                    go.Candlestick(
                        x=dados.index,
                        open=dados["Open"],
                        high=dados["High"],
                        low=dados["Low"],
                        close=dados["Close"],
                        name="Candle",
                    )
                ]
            )

            # Adicionar as médias móveis
            fig.add_trace(
                go.Scatter(
                    x=dados.index,
                    y=dados["MA20"],
                    mode="lines",
                    line=dict(color="blue", width=1),
                    name="Média Móvel de 20 períodos",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=dados.index,
                    y=dados["MA50"],
                    mode="lines",
                    line=dict(color="red", width=1),
                    name="Média Móvel de 50 períodos",
                )
            )

            # Adicionar título e rótulos aos eixos
            fig.update_layout(
                title="Gráfico de Candlestick", xaxis_title="Data", yaxis_title="Preço"
            )

            # Exibir o gráfico no Streamlit
            st.plotly_chart(fig)

            # Calcular os retornos diários
            dados["Daily Returns"] = dados["Close"].pct_change()

            # Calcular a volatilidade (desvio padrão dos retornos diários)
            volatilidade = dados["Daily Returns"].rolling(window=30).std()

            # Criar o gráfico
            fig = px.line(
                dados,
                x="Date",
                y="Close",
                color="Nome",
                title="Preço vs Volatilidade",
                labels={"Date": "Data", "Close": " R$ Preço"},
            )
            fig.add_scatter(
                x=dados["Date"], y=volatilidade, name="Volatilidade", yaxis="y2"
            )

            # Atualizar layout com eixo y secundário para a volatilidade
            fig.update_layout(
                yaxis2=dict(title="Volatilidade", overlaying="y", side="right")
            )
            # Exibir o gráfico
            st.plotly_chart(fig)

    else:
        st.write("Selecione as ações e o periodo que você quer analisar⬆️")

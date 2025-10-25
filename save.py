# app.py
import streamlit as st
import joblib
import pandas as pd
import re

# Configuração da página
st.set_page_config(page_title="Detector de Fake News", page_icon="🤖", layout="centered")
st.title("🤖 Detector de Fake News em Português")
st.write("""
Este é um protótipo de IA treinado para identificar notícias potencialmente falsas.
Digite ou cole o texto da notícia abaixo para ver a análise.
""")

# Função para pré-processar o texto (IGUAL à usada no treinamento)
def preprocessar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'[^a-záàâãéèêíïóôõöúçñ ]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

# Carregar o modelo e o vectorizer (USANDO OS ARQUIVOS QUE VOCÊS SALVARAM)
@st.cache_resource
def carregar_modelos():
    try:
        model = joblib.load('modelo_svm.joblib')
        vectorizer = joblib.load('vectorizer.joblib')
        return model, vectorizer
    except Exception as e:
        st.error(f"Erro ao carregar modelos: {e}")
        return None, None

model, vectorizer = carregar_modelos()

# Se os modelos não carregarem, não exibe o restante
if model is None or vectorizer is None:
    st.stop()

# Área de input do usuário
texto_usuario = st.text_area(
    label="**Cole o texto da notícia aqui:**",
    height=200,
    placeholder="Ex: 'Cientistas descobrem que comer chocolate todo dia emagrece...'",
    help="Quanto mais texto, melhor será a análise."
)

# Botão para análise
if st.button("Analisar Notícia 🔍", type="primary", use_container_width=True):
    if texto_usuario.strip() == "":
        st.warning("Por favor, insira um texto para análise.")
    else:
        with st.spinner("Analisando o texto..."):
            # Pré-processa o texto
            texto_limpo = preprocessar_texto(texto_usuario)
            
            # Vetoriza o texto (transforma em números)
            texto_vetorizado = vectorizer.transform([texto_limpo])
            
            # Faz a previsão
            previsao = model.predict(texto_vetorizado)
            probabilidade = model.predict_proba(texto_vetorizado)
            
            # Mostra os resultados
            st.subheader("📊 Resultado da Análise:")
            
            if previsao[0] == 0:
                st.error(f"**🚫 POTENCIAL FAKE NEWS**")
                st.progress(probabilidade[0][0])
                st.write(f"**Confiança da análise:** {probabilidade[0][0] * 100:.2f}%")
            else:
                st.success(f"**✅ NOTÍCIA CONFIÁVEL**")
                st.progress(probabilidade[0][1])
                st.write(f"**Confiança da análise:** {probabilidade[0][1] * 100:.2f}%")
            
            # Dashboard de informações
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Probabilidade de ser Fake", f"{probabilidade[0][0] * 100:.2f}%")
            with col2:
                st.metric("Probabilidade de ser Verdadeira", f"{probabilidade[0][1] * 100:.2f}%")

# Informações sobre o projeto
with st.expander("ℹ️ Sobre este projeto"):
    st.write("""
    **Tecnologia:** 
    - Modelo de Machine Learning (SVM) treinado com o dataset Fake.Br Corpus
    - Acuracia do modelo: **96.81%**
    - Processamento de Linguagem Natural (NLP)
    
    **Como funciona:**
    O modelo analisa padrões linguísticos e palavras-chave presentes em notícias 
    previamente classificadas como verdadeiras ou falsas.
    
    **Desenvolvido por:** [Nomes do Grupo]
    """)

# Rodapé
st.markdown("---")
st.caption("⚠️ Este é um protótipo para fins educacionais. Sempre verifique informações em fontes confiáveis.")
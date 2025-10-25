# app.py
import streamlit as st
import joblib
import pandas as pd
import re

# Configuração da página
st.set_page_config(page_title="Análise de credibilidade de Fake News", page_icon="🤖", layout="centered")
st.title("🤖 Análise de Credibilidade de Fake News em Português")
st.write("""
Este é um protótipo de IA treinado para analisar a correspondência de notícias potencialmente falsas.
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
            
            # Faz a previsão (probabilidades)
            try:
                probabilidade = model.predict_proba(texto_vetorizado)
            except Exception as e:
                st.error(f"Erro ao prever (predict_proba): {e}")
                st.stop()
            
            # Determinar índice da classe 'real' (assume que label real == 1)
            try:
                classes = list(model.classes_)
                # tenta encontrar o índice da classe '1' (notícia verdadeira)
                if 1 in classes:
                    idx_real = classes.index(1)
                else:
                    # se não existir o label 1, assume que a classe com maior média de probabilidade corresponde a "real" (fallback)
                    idx_real = 1 if len(classes) > 1 else 0
            except Exception:
                idx_real = 1  # fallback simples
            
            real_prob = float(probabilidade[0][idx_real])
            fake_prob = 1.0 - real_prob  # binário esperado
            
            # Mostra os resultados com sua "trava"
            st.subheader("📊 Resultado da Análise:")
            
            # Regras de classificação com 'trava' entre 50% e 70%
            if fake_prob > 0.8:
                st.error("**🚫 POTENCIAL FAKE NEWS**")
                st.progress(real_prob)  # mostra barra com prob do real (vai estar baixa)
            elif 0.5 <= fake_prob < 0.8:
                st.warning("**⚠️ POTENCIALMENTE REAL — verificar manualmente**")
                st.progress(real_prob)
                st.info("Este texto está na faixa 50%–80% — classificado como potencialmente real (recomendado: checar fontes).")
            else:
                st.success("**✅ ALTA CONFIABILIDADE**")
                st.progress(real_prob)
                
            # Dashboard de informações
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Correspondência com notícia Fake", f"{fake_prob * 100:.2f}%")
            with col2:
                st.metric("Correspondência com notícia Verdadeira", f"{real_prob * 100:.2f}%")

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
    
    **Desenvolvido por:** Veritas-SP
    """)

# Rodapé
st.markdown("---")
st.caption("⚠️ Este é um protótipo para fins educacionais. Sempre verifique informações em fontes confiáveis.")


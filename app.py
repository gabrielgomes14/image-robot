import streamlit as st
from inference_sdk import InferenceHTTPClient
import base64
import os
from dotenv import load_dotenv

# Configurar cliente Roboflow
import os

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.getenv("ROBOFLOW_API_KEY")
)

st.title("Upload e Análise de Imagem com Roboflow")

uploaded_file = st.file_uploader("Envie uma imagem", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Mostrar preview da imagem
    st.image(uploaded_file, caption="Imagem enviada", use_column_width=True)

    # Ler imagem e converter para base64 para envio
    img_bytes = uploaded_file.read()
    encoded_string = base64.b64encode(img_bytes).decode("utf-8")
    base64_image = f"data:image/jpeg;base64,{encoded_string}"

    # Botão para enviar imagem ao workflow
    if st.button("Enviar para análise"):
        with st.spinner("Processando..."):
            try:
                result = client.run_workflow(
                    workspace_name="teste-yrohx",
                    workflow_id="custom-workflow",
                    images={"image": base64_image},
                    use_cache=True
                )
                st.success("Análise concluída!")

                # Tenta mostrar campo 'open_ai'
                st.subheader("Resumo gerado por IA:")
                try:
                    openai_output = result[0]["open_ai"]
                    st.markdown(f"> {openai_output}")
                except (KeyError, IndexError, TypeError):
                    st.warning("Não foi possível encontrar o campo 'open_ai' no resultado.")

            except Exception as e:
                st.error(f"Erro ao processar a imagem: {e}")

# app.py

import streamlit as st
import pandas as pd
import io
from datetime import datetime
import numpy as np

# Importar los módulos refactorizados
import modulo_0st as m0
import modulo_1st as m1
import modulo_2st2 as m2
import modulo_3st as m3
import prompt_creator_app_st as prompt_creator

# --- Configuración de la Aplicación ---
APP_TITLE = "Analizador Estadístico de Encuestas de Residentes"
APP_VERSION = "1.0"
CORRECT_PASSWORD = "fse2025" # Contraseña de acceso

st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Función de Autenticación ---
def check_password():
    """Returns `True` if the user enters the correct password."""
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if not st.session_state["password_correct"]:
        st.sidebar.empty() # Limpiar contenido de la barra lateral antes de iniciar sesión
        st.title("Acceso Requerido")
        st.write("Por favor, introduce la contraseña para acceder a la aplicación.")
        
        password_input = st.text_input("Contraseña", type="password")
        
        if st.button("Acceder"):
            if password_input == CORRECT_PASSWORD:
                st.session_state["password_correct"] = True
                st.rerun() # Usar st.rerun() en lugar de st.experimental_rerun()
            else:
                st.error("😕 Contraseña incorrecta. Inténtalo de nuevo.")
        return False
    return True

# --- Lógica principal de la aplicación ---
# Muestra la aplicación solo si la contraseña es correcta
if check_password():
    st.sidebar.title(APP_TITLE)
    st.sidebar.subheader(f"Versión {APP_VERSION}")

    # Pestañas de la aplicación
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Carga de Datos", 
        "Análisis Cuantitativo", 
        "Análisis Cualitativo", 
        "Análisis de Comparación", 
        "Creador de Prompts IA"
    ])

    with tab1:
        st.header("1️⃣ Carga y Procesamiento de Archivos")
        st.markdown("Carga uno o más archivos de Excel (`.xlsx`) para comenzar el análisis.")
        uploaded_files = st.file_uploader(
            "Arrastra y suelta tus archivos aquí (o haz clic para buscar)", 
            type=["xlsx"], 
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if 'df_global' not in st.session_state or 'uploaded_files_cache' not in st.session_state or sorted([f.name for f in uploaded_files]) != sorted([f.name for f in st.session_state['uploaded_files_cache']]):
                st.session_state['uploaded_files_cache'] = uploaded_files
                st.session_state['df_global'] = None
                
            # CORRECCIÓN: Llamar a la función con el nombre correcto
            st.session_state['df_global'] = m0.process_uploaded_files_concurrently(uploaded_files, st)
            
            if st.session_state['df_global'] is not None and not st.session_state['df_global'].empty:
                st.subheader("Vista previa de los datos procesados:")
                st.dataframe(st.session_state['df_global'].head())
                st.info(f"Datos cargados exitosamente. Total de filas: {len(st.session_state['df_global'])}.")
            
    if 'df_global' in st.session_state and st.session_state['df_global'] is not None:
        
        with tab2:
            st.header("2️⃣ Análisis Cuantitativo")
            st.markdown("Realiza un análisis estadístico de las columnas numéricas y de selección múltiple.")
            
            # Columnas numéricas
            numeric_cols = st.session_state['df_global'].select_dtypes(include=np.number).columns.tolist()
            # Columnas de selección múltiple (asumimos que tienen menos de 10 valores únicos y no son numéricas continuas)
            categorical_cols = [col for col in st.session_state['df_global'].columns if col not in numeric_cols and st.session_state['df_global'][col].nunique() < 20]
            
            all_cols_analysis = sorted(numeric_cols + categorical_cols)
            
            if not all_cols_analysis:
                st.warning("No se encontraron columnas adecuadas para el análisis cuantitativo. Por favor, revisa tus datos.")
            else:
                selected_cols = st.multiselect("Selecciona las columnas para el análisis:", all_cols_analysis, key="cuanti_cols")
                
                analysis_type = st.radio("Tipo de Análisis para columnas de texto/selección:", 
                                        ["Análisis de Frecuencia"], 
                                        key="cuanti_type", horizontal=True)
                
                graph_style = st.selectbox("Estilo de Gráfico:", 
                                        ["Barras", "Circular", "Histograma"], 
                                        key="cuanti_graph")
                                        
                if st.button("Ejecutar Análisis Cuantitativo", key="run_cuanti_button") and selected_cols:
                    results_sequence = m1.perform_quantitative_analysis(st.session_state['df_global'], selected_cols, analysis_type, graph_style, st)
                    
                    st.session_state['quantitative_results'] = results_sequence
            
            if 'quantitative_results' in st.session_state and st.session_state['quantitative_results']:
                st.subheader("Resultados del Análisis:")
                for item_type, item_title, item_content in st.session_state['quantitative_results']:
                    if item_type == 'text':
                        st.markdown(item_content)
                    elif item_type == 'image_bytes':
                        st.image(item_content, caption=item_title)
            
                # Botón de descarga
                doc_bytes = m1.export_to_word(st.session_state['quantitative_results'])
                st.download_button(
                    label="📥 Descargar Documento (Word)",
                    data=doc_bytes,
                    file_name=f"Analisis_Cuantitativo_{datetime.now().strftime('%Y%m%d')}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )

        with tab3:
            st.header("3️⃣ Análisis Cualitativo y de Texto")
            st.markdown("Realiza un análisis de texto en las columnas con respuestas abiertas.")

            # Filtrar columnas de texto
            text_columns = st.session_state['df_global'].select_dtypes(include=['object']).columns.tolist()
            text_columns_filtered = [col for col in text_columns if st.session_state['df_global'][col].nunique() > 20]
            
            if not text_columns_filtered:
                st.warning("No se encontraron columnas de texto adecuadas (con más de 20 valores únicos) para el análisis. Por favor, revisa tus datos.")
            else:
                selected_text_cols = st.multiselect("Selecciona las columnas de texto para el análisis:", text_columns_filtered, key="cualitativo_cols")
                
                analysis_options = {
                    "word_freq": st.checkbox("Análisis de Frecuencia de Palabras y N-gramas", value=True),
                    "sentiment_analysis": st.checkbox("Análisis de Sentimiento (deshabilitado)", value=False, disabled=True),
                    "topic_modeling": st.checkbox("Modelado de Temas (NMF o LDA)", value=True)
                }
                
                if analysis_options.get('topic_modeling'):
                    topic_method = st.radio("Método de modelado de temas:", ["NMF", "LDA"], horizontal=True)
                    num_topics = st.slider("Número de temas a extraer:", 2, 10, 5)
                    analysis_options['num_topics'] = num_topics
                    analysis_options['topic_method'] = topic_method
                
                # Campos para la IA
                st.subheader("Interacción con IA (opcional)")
                ai_model_choice = st.selectbox(
                    "Selecciona el modelo de IA:",
                    ["Ninguno", "Gemini Pro", "GPT-3.5", "Claude 3 Sonnet"],
                    index=0,
                    disabled=True,
                    help="Modelos de IA deshabilitados temporalmente para evitar costos."
                )
                ai_prompt = st.text_area(
                    "Introduce el prompt para la IA (ej. 'Resume los puntos principales'):",
                    height=100,
                    disabled=ai_model_choice == "Ninguno"
                )

                if st.button("Ejecutar Análisis Cualitativo", key="run_cualitativo_button") and selected_text_cols:
                    if not any(analysis_options.values()):
                        st.error("Por favor, selecciona al menos una opción de análisis.")
                    else:
                        st.session_state['qualitative_results'] = m2.perform_qualitative_analysis(st.session_state['df_global'], selected_text_cols, analysis_options, ai_model_choice, ai_prompt, st)
                
            if 'qualitative_results' in st.session_state and st.session_state['qualitative_results']:
                st.subheader("Resultados del Análisis:")
                for item_type, item_title, item_content in st.session_state['qualitative_results']:
                    if item_type == 'text':
                        st.markdown(item_content)
                    elif item_type == 'image_bytes':
                        st.image(item_content, caption=item_title)
            
                doc_bytes = m2.export_qualitative_to_word(st.session_state['qualitative_results'])
                st.download_button(
                    label="📥 Descargar Documento (Word)",
                    data=doc_bytes,
                    file_name=f"Analisis_Cualitativo_{datetime.now().strftime('%Y%m%d')}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )

        with tab4:
            st.header("4️⃣ Análisis de Comparación")
            st.markdown("Compara las respuestas a una pregunta de encuesta en función de una o más variables de agrupación (ej. 'Edad' o 'Unidad').")
            
            all_cols = st.session_state['df_global'].columns.tolist()
            
            if not all_cols:
                 st.warning("No se encontraron columnas para realizar el análisis de comparación. Por favor, revisa tus datos.")
            else:
                crit_col = st.selectbox(
                    "Selecciona la columna que define los grupos de comparación:",
                    all_cols,
                    key="crit_col"
                )
                
                # Filtra las columnas de preguntas para que no incluyan la columna de criterio
                question_cols = [col for col in all_cols if col != crit_col]
                selected_question_cols = st.multiselect(
                    "Selecciona las preguntas a analizar por grupo:",
                    question_cols,
                    key="question_cols"
                )
                
                if st.button("Ejecutar Comparación", key="run_comparison_button") and crit_col and selected_question_cols:
                    st.session_state['comparison_results'] = m3.perform_comparison_analysis(st.session_state['df_global'], crit_col, selected_question_cols, st)
            
            if 'comparison_results' in st.session_state and st.session_state['comparison_results']:
                st.subheader("Resultados de la Comparación:")
                for item_type, item_title, item_content in st.session_state['comparison_results']:
                    if item_type == 'text':
                        st.markdown(item_content)
                    elif item_type == 'image_bytes':
                        st.image(item_content, caption=item_title)

                doc_bytes = m3.export_comparison_to_word(st.session_state['comparison_results'], {})
                st.download_button(
                    label="📥 Descargar Documento (Word)",
                    data=doc_bytes,
                    file_name=f"Analisis_Comparacion_{datetime.now().strftime('%Y%m%d')}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
        
        with tab5:
            st.header("5️⃣ Creador de Prompts para IA")
            st.write("Utiliza esta herramienta para construir prompts estructurados para modelos de Inteligencia Artificial.")

            verbs = prompt_creator.get_all_verbs()
            role_categories = prompt_creator.get_role_categories()

            col_verb_standalone, col_cat_standalone = st.columns(2)
            with col_verb_standalone:
                selected_verb_standalone = st.selectbox("Verbo de Acción:", verbs, key="prompt_verb_standalone")
            with col_cat_standalone:
                selected_category_standalone = st.selectbox("Categoría de Rol:", role_categories, key="prompt_category_standalone")
            
            roles_standalone = prompt_creator.get_roles_by_category(selected_category_standalone)
            selected_role_standalone = st.selectbox("Rol Específico:", roles_standalone, key="prompt_role_standalone")
            
            additional_context_standalone = st.text_area("Contexto/Instrucciones Adicionales:", key="prompt_context_standalone")
            
            generated_prompt_final = prompt_creator.generate_ai_prompt(selected_verb_standalone, selected_role_standalone, additional_context_standalone)
            
            st.subheader("Prompt Generado:")
            st.text_area("Copia este prompt para usarlo en tus interacciones con IA:", value=generated_prompt_final, height=200, key="final_generated_prompt_display")

            st.info("Para copiar el prompt, selecciona el texto en el cuadro de arriba y usa Ctrl+C (o Cmd+C en Mac).")
            st.warning("Esta herramienta es experimental. Siempre revisa y ajusta los prompts generados según tus necesidades.")

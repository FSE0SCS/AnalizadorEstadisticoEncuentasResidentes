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
                st.rerun()
            else:
                st.error("Contraseña incorrecta. Por favor, inténtalo de nuevo.")
        
        return st.session_state["password_correct"]

# --- Lógica principal de la aplicación ---
if check_password():
    
    # --- Barra Lateral ---
    st.sidebar.title("🛠️ Opciones")
    st.sidebar.markdown(f"**{APP_TITLE}** v{APP_VERSION}")

    # Selección de módulo
    module_choice = st.sidebar.radio(
        "Elige un módulo:",
        ["1. Carga y Procesamiento", "2. Análisis Descriptivo", "3. Análisis de Texto", "4. Análisis Comparativo", "5. Creador de Prompts"],
        index=0 # Inicia en el primer módulo
    )
    st.sidebar.markdown("---")

    if module_choice == "1. Carga y Procesamiento":
        st.title("Módulo 1: Carga y Procesamiento de Datos")
        st.write("Sube uno o varios archivos de Excel para consolidarlos en un único DataFrame para su análisis.")
        
        uploaded_files = st.file_uploader(
            "Selecciona archivos de Excel",
            type=['xlsx', 'xls'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button("💾 Procesar y Consolidar Datos"):
                st.session_state['results_df'], status, message = m0.process_excel_files(uploaded_files)
                if status == "success":
                    st.session_state['processing_summary'] = message
                else:
                    st.session_state['processing_summary'] = message

        if 'results_df' in st.session_state and st.session_state['results_df'] is not None:
            st.subheader("Datos Consolidado:")
            st.write(st.session_state['results_df'])
            
            # Botón de descarga
            excel_data = io.BytesIO()
            st.session_state['results_df'].to_excel(excel_data, index=False, engine='openpyxl')
            excel_data.seek(0)
            st.download_button(
                label="📥 Descargar Datos Consolidado como Excel",
                data=excel_data,
                file_name=f"datos_consolidados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            if 'processing_summary' in st.session_state:
                st.markdown("---")
                st.subheader("Resumen del Procesamiento:")
                st.info(st.session_state['processing_summary'])

    elif module_choice == "2. Análisis Descriptivo":
        st.title("Módulo 2: Análisis Descriptivo Cuantitativo")
        st.write("Selecciona una o más columnas para realizar un análisis descriptivo y generar gráficos.")
        
        if 'results_df' in st.session_state and st.session_state['results_df'] is not None:
            df = st.session_state['results_df']
            
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            text_cols = df.select_dtypes(include='object').columns.tolist()
            
            analysis_type = st.radio("Tipo de Análisis", ["Cuantitativo", "Frecuencia"], horizontal=True)
            
            if analysis_type == "Cuantitativo":
                selected_cols = st.multiselect("Selecciona columnas numéricas:", numeric_cols)
                graph_style = st.selectbox("Estilo de Gráfico:", ["Histograma", "Gráfico de Caja (Boxplot)"])
                if selected_cols and st.button("📊 Generar Análisis"):
                    if 'analysis_results' not in st.session_state:
                        st.session_state['analysis_results'] = []
                    
                    st.session_state['analysis_results'] = m1.perform_quantitative_analysis(df, selected_cols, graph_style)

            else: # Frecuencia
                selected_cols = st.multiselect("Selecciona columnas para el análisis de frecuencia:", text_cols)
                if selected_cols and st.button("📊 Generar Análisis de Frecuencia"):
                    if 'analysis_results' not in st.session_state:
                        st.session_state['analysis_results'] = []
                    
                    st.session_state['analysis_results'] = m1.perform_frequency_analysis(df, selected_cols)
            
            if 'analysis_results' in st.session_state and st.session_state['analysis_results']:
                st.markdown("---")
                st.subheader("Resultados del Análisis:")
                
                # Mostrar los resultados secuencialmente
                for item_type, title, content in st.session_state['analysis_results']:
                    if item_type == 'text':
                        st.text(content)
                    elif item_type == 'image_bytes':
                        st.image(content, caption=title)
                
                # Exportar a Word
                word_doc_bytes = m1.export_analysis_to_word(st.session_state['analysis_results'])
                st.download_button(
                    label="📥 Descargar Informe en Word",
                    data=word_doc_bytes,
                    file_name=f"informe_descriptivo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )

        else:
            st.warning("Por favor, carga y procesa los datos en el 'Módulo 1' primero.")
            
    elif module_choice == "3. Análisis de Texto":
        st.title("Módulo 3: Análisis de Texto")
        st.write("Analiza y extrae información de columnas de texto.")
        
        if 'results_df' in st.session_state and st.session_state['results_df'] is not None:
            df = st.session_state['results_df']
            
            text_columns = df.select_dtypes(include='object').columns.tolist()
            selected_text_cols = st.multiselect("Selecciona columnas de texto para analizar:", text_columns)
            
            analysis_options = {}
            st.subheader("Opciones de Análisis:")
            analysis_options['word_frequency'] = st.checkbox("Análisis de Frecuencia de Palabras y N-gramas", value=True)
            analysis_options['sentiment_analysis'] = st.checkbox("Análisis de Sentimiento", value=True)
            analysis_options['topic_modeling'] = st.checkbox("Modelado de Temas (NMF)", value=False)
            if analysis_options['topic_modeling']:
                analysis_options['num_topics'] = st.slider("Número de Temas", min_value=2, max_value=10, value=5)
            
            st.markdown("---")
            st.subheader("Opciones de IA (Experimental)")
            enable_ai = st.checkbox("Habilitar Análisis con IA (Gemini)")
            if enable_ai:
                ai_model = st.selectbox("Selecciona un modelo de IA:", ["gemini-2.5-flash-preview-05-20"])
                ai_prompt = st.text_area("Ingresa un prompt para la IA (ej: 'Resume los temas principales de este texto'):")
                analysis_options['ai_model'] = ai_model
                analysis_options['ai_prompt'] = ai_prompt

            if selected_text_cols and st.button("🔬 Realizar Análisis de Texto"):
                with st.spinner("Analizando texto..."):
                    st.session_state['text_analysis_results'] = m2.perform_text_analysis(df, selected_text_cols, analysis_options)
            
            if 'text_analysis_results' in st.session_state and st.session_state['text_analysis_results']:
                st.markdown("---")
                st.subheader("Resultados del Análisis de Texto:")
                for item_type, title, content in st.session_state['text_analysis_results']:
                    if item_type == 'text':
                        st.markdown(content)
                    elif item_type == 'image_bytes':
                        st.image(content, caption=title)
                
                word_doc_bytes = m2.export_analysis_to_word(st.session_state['text_analysis_results'])
                st.download_button(
                    label="📥 Descargar Informe de Texto en Word",
                    data=word_doc_bytes,
                    file_name=f"informe_texto_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )

        else:
            st.warning("Por favor, carga y procesa los datos en el 'Módulo 1' primero.")
            
    elif module_choice == "4. Análisis Comparativo":
        st.title("Módulo 4: Análisis Comparativo")
        st.write("Compara la distribución de una pregunta (variable) en función de un criterio (otra variable).")
        
        if 'results_df' in st.session_state and st.session_state['results_df'] is not None:
            df = st.session_state['results_df']
            
            available_cols = df.columns.tolist()
            
            crit_col = st.selectbox("Selecciona la columna de criterio (variable de agrupamiento):", available_cols)
            
            st.markdown("---")
            st.subheader("Preguntas a comparar")
            question_cols = st.multiselect("Selecciona las preguntas a comparar:", [col for col in available_cols if col != crit_col])
            
            if crit_col and question_cols and st.button("📈 Generar Análisis Comparativo"):
                with st.spinner("Generando análisis comparativo..."):
                    st.session_state['comparison_results'] = m3.perform_comparison_analysis(df, crit_col, question_cols)
            
            if 'comparison_results' in st.session_state and st.session_state['comparison_results']:
                st.markdown("---")
                st.subheader("Resultados del Análisis Comparativo:")
                for item_type, title, content in st.session_state['comparison_results']:
                    if item_type == 'text':
                        st.markdown(content)
                    elif item_type == 'image_bytes':
                        st.image(content, caption=title)
                
                word_doc_bytes = m3.export_comparison_to_word(st.session_state['comparison_results'])
                st.download_button(
                    label="📥 Descargar Informe Comparativo en Word",
                    data=word_doc_bytes,
                    file_name=f"informe_comparativo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
        else:
            st.warning("Por favor, carga y procesa los datos en el 'Módulo 1' primero.")
            
    elif module_choice == "5. Creador de Prompts":
        st.title("Módulo 5: Creador de Prompts para IA")
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


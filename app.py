# app.py
# Este es el archivo principal de la aplicación Streamlit.

import streamlit as st
import pandas as pd
import io
from datetime import datetime

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
st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")

# --- Función de Autenticación ---
def check_password():
    """
    Controla el acceso a la aplicación mediante una contraseña.
    Retorna True si el usuario ingresa la contraseña correcta.
    """
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
                st.experimental_rerun() # Volver a ejecutar la app para mostrar el contenido
            else:
                st.error("Contraseña incorrecta. Por favor, inténtalo de nuevo.")
        return False
    return True

# --- Lógica principal de la aplicación ---
if check_password():
    st.title(f"{APP_TITLE} (v{APP_VERSION})")
    
    # === BARRA LATERAL PARA CARGA Y CONFIGURACIÓN ===
    st.sidebar.header("1. Carga de Datos")
    uploaded_files = st.sidebar.file_uploader(
        "Selecciona uno o más archivos Excel (.xlsx)", 
        type=["xlsx"], 
        accept_multiple_files=True
    )

    df_final = None
    if uploaded_files:
        if st.sidebar.button("Cargar y Procesar"):
            st.session_state["analysis_ready"] = False
            with st.spinner("Procesando archivos..."):
                progress_bar = st.progress(0)
                df_final, status, message = m0.process_multiple_files(
                    uploaded_files, 
                    lambda p, msg: progress_bar.progress(p, text=msg),
                    m0._log_message_streamlit
                )
            if status == "success":
                st.session_state["df_final"] = df_final
                st.session_state["analysis_ready"] = True
                st.success(message)
            else:
                st.error(message)

    if "analysis_ready" in st.session_state and st.session_state["analysis_ready"]:
        df_final = st.session_state["df_final"]
        st.sidebar.success("Archivos cargados y listos para analizar.")
        st.sidebar.write(f"Filas totales: {len(df_final):,}")
        st.sidebar.write(f"Columnas totales: {len(df_final.columns)}")
        
        # Opciones globales de análisis
        numeric_cols = [col for col in df_final.columns if pd.api.types.is_numeric_dtype(df_final[col])]
        text_cols = [col for col in df_final.columns if pd.api.types.is_string_dtype(df_final[col])]
        
        st.sidebar.markdown("---")
        st.sidebar.header("Opciones de Análisis")
        st.sidebar.subheader("Análisis Cuantitativo")
        selected_numeric_cols = st.sidebar.multiselect(
            "Selecciona columnas numéricas:", 
            numeric_cols
        )
        numeric_chart_type = st.sidebar.selectbox(
            "Tipo de gráfico cuantitativo:", 
            ['histogram', 'boxplot']
        )
        
        st.sidebar.subheader("Análisis Textual")
        selected_text_cols = st.sidebar.multiselect(
            "Selecciona columnas de texto:", 
            text_cols
        )
        if 'analysis_options' not in st.session_state: st.session_state['analysis_options'] = {}
        st.session_state['analysis_options']['topic_modeling'] = st.sidebar.checkbox("Modelado de Temas (NMF/LDA)", value=False)
        st.session_state['analysis_options']['sentiment_analysis'] = st.sidebar.checkbox("Análisis de Sentimiento", value=False)
        st.session_state['analysis_options']['word_frequency'] = st.sidebar.checkbox("Frecuencia de Palabras y N-gramas", value=False)
        
        st.sidebar.subheader("Análisis Comparativo")
        selected_comparison_criteria = st.sidebar.multiselect(
            "Selecciona criterios de comparación (e.g., por Unidad):",
            [col for col in df_final.columns if len(df_final[col].unique()) < 50],
            key="comparison_criteria"
        )
        selected_comparison_questions = st.sidebar.multiselect(
            "Selecciona preguntas a comparar:",
            [col for col in df_final.columns if col not in selected_comparison_criteria],
            key="comparison_questions"
        )

    # === TABS PRINCIPALES PARA LA NAVEGACIÓN ===
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Procesamiento", 
        "Análisis Cuantitativo", 
        "Análisis Textual", 
        "Análisis Comparativo", 
        "Generador de Prompts"
    ])

    with tab1:
        st.header("Vista Previa del DataFrame")
        if "df_final" in st.session_state and not st.session_state["df_final"].empty:
            st.write(f"Se muestran las primeras {len(st.session_state['df_final'])} filas del DataFrame procesado.")
            st.dataframe(st.session_state["df_final"])
        else:
            st.info("Por favor, carga y procesa los archivos en la barra lateral.")

    with tab2:
        st.header("Análisis Cuantitativo")
        if "analysis_ready" in st.session_state and st.session_state["analysis_ready"]:
            if st.button("Ejecutar Análisis Cuantitativo"):
                if not selected_numeric_cols:
                    st.warning("Por favor, selecciona al menos una columna numérica en la barra lateral.")
                else:
                    analysis_options = {'numeric_chart_type': numeric_chart_type}
                    doc_bytes, status, message = m1.perform_quantitative_analysis(
                        df_final, 
                        selected_numeric_cols,
                        analysis_options
                    )
                    if status == "success":
                        st.session_state["quant_doc_bytes"] = doc_bytes
                        st.success(message)
            
            if "quant_doc_bytes" in st.session_state:
                st.download_button(
                    label="Descargar Informe Cuantitativo (.docx)",
                    data=st.session_state["quant_doc_bytes"],
                    file_name="informe_analisis_cuantitativo.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )

    with tab3:
        st.header("Análisis Textual")
        if "analysis_ready" in st.session_state and st.session_state["analysis_ready"]:
            if st.button("Ejecutar Análisis Textual"):
                if not selected_text_cols:
                    st.warning("Por favor, selecciona al menos una columna de texto en la barra lateral.")
                else:
                    doc_bytes, status, message = m2.perform_textual_analysis(
                        df_final, 
                        selected_text_cols,
                        st.session_state['analysis_options']
                    )
                    if status == "success":
                        st.session_state["text_doc_bytes"] = doc_bytes
                        st.success(message)
            
            if "text_doc_bytes" in st.session_state:
                st.download_button(
                    label="Descargar Informe Textual (.docx)",
                    data=st.session_state["text_doc_bytes"],
                    file_name="informe_analisis_textual.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )

    with tab4:
        st.header("Análisis Comparativo")
        if "analysis_ready" in st.session_state and st.session_state["analysis_ready"]:
            if st.button("Ejecutar Análisis Comparativo"):
                if not selected_comparison_criteria or not selected_comparison_questions:
                    st.warning("Por favor, selecciona al menos un criterio y una pregunta de comparación.")
                else:
                    doc_bytes, status, message = m3.perform_comparative_analysis(
                        df_final, 
                        selected_comparison_criteria, 
                        selected_comparison_questions
                    )
                    if status == "success":
                        st.session_state["comp_doc_bytes"] = doc_bytes
                        st.success(message)
            
            if "comp_doc_bytes" in st.session_state:
                st.download_button(
                    label="Descargar Informe Comparativo (.docx)",
                    data=st.session_state["comp_doc_bytes"],
                    file_name="informe_analisis_comparativo.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )

    with tab5:
        st.header("Generador de Prompts para IA")
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
        st.text_area(
            "Copia este prompt para usarlo en tus interacciones con IA:", 
            value=generated_prompt_final, 
            height=200, 
            key="final_generated_prompt_display"
        )
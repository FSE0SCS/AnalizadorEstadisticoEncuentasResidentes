# app.py

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
                st.rerun() # Volver a ejecutar la app para mostrar el contenido
            else:
                st.error("Contraseña incorrecta. Inténtalo de nuevo.")
        return False
    return True

# --- Lógica Principal de la Aplicación (después de autenticación) ---
if check_password():
    st.sidebar.title("Navegación")
    st.sidebar.markdown(f"**{APP_TITLE}**")
    st.sidebar.markdown(f"Versión: `{APP_VERSION}`")
    st.sidebar.markdown("---")

    app_choice = st.sidebar.radio(
        "Selecciona una Herramienta:",
        ["Fusionador de Excel", "Análisis Cuantitativo Básico", "Análisis Cuantitativo y Comparativo", "Análisis Cualitativo (PLN + IA)", "Creador de Prompts IA"]
    )

    st.title(APP_TITLE)
    st.markdown("---")

    # --- Sección: Fusionador de Excel ---
    if app_choice == "Fusionador de Excel":
        st.header("📁 Fusionador de Excel Ultra Rápido")
        st.write("Carga múltiples archivos Excel para fusionarlos en uno solo. Ideal para combinar datos de encuestas de diferentes fuentes.")

        uploaded_files = st.file_uploader(
            "Arrastra y suelta tus archivos Excel aquí (o haz clic para buscar)",
            type=["xlsx", "xls"],
            accept_multiple_files=True,
            key="excel_merger_uploader"
        )

        if uploaded_files:
            st.info(f"Archivos cargados: {len(uploaded_files)}")
            for file in uploaded_files:
                st.markdown(f"- `{file.name}`")

            if st.button("⚡ Procesar y Fusionar", key="merge_button"):
                # Convertir los UploadedFile a BytesIO para que el módulo los pueda leer
                file_contents_for_merge = [io.BytesIO(file.read()) for file in uploaded_files]
                
                with st.spinner("Fusionando archivos, esto puede tardar..."):
                    merged_df, status_level, message = m0.merge_excel_files_streamlit(file_contents_for_merge)
                    
                    if status_level == "success":
                        st.success(message)
                        st.subheader("Primeras 5 filas del DataFrame Fusionado:")
                        st.dataframe(merged_df.head())
                        
                        # Botón de descarga para CSV
                        csv_output = merged_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="💾 Descargar CSV Fusionado",
                            data=csv_output,
                            file_name=f"fusion_excel_resultado_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            key="download_csv_merged"
                        )
                        
                        # Botón de descarga para Excel
                        excel_output = io.BytesIO()
                        with pd.ExcelWriter(excel_output, engine='openpyxl') as writer:
                            merged_df.to_excel(writer, sheet_name='Hoja1', index=False, header=False)
                        excel_output.seek(0)
                        st.download_button(
                            label="💾 Descargar Excel Fusionado",
                            data=excel_output.getvalue(),
                            file_name=f"fusion_excel_resultado_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.document",
                            key="download_excel_merged"
                        )
                    else:
                        st.error(message)
        else:
            st.info("Por favor, sube uno o más archivos Excel para comenzar la fusión.")

    # --- Sección: Análisis Cuantitativo Básico ---
    elif app_choice == "Análisis Cuantitativo Básico":
        st.header("📊 Análisis Cuantitativo Básico")
        st.write("Carga un archivo Excel para realizar análisis descriptivos básicos y generar gráficos para columnas seleccionadas.")
        
        uploaded_file_basic_analysis = st.file_uploader(
            "Sube tu archivo Excel (.xlsx) para análisis cuantitativo",
            type=["xlsx"],
            key="basic_analysis_uploader"
        )

        if uploaded_file_basic_analysis:
            try:
                df_basic = pd.read_excel(uploaded_file_basic_analysis)
                st.subheader("Vista Previa de los Datos:")
                st.dataframe(df_basic.head())
                st.write(f"Columnas detectadas: `{', '.join(df_basic.columns.tolist())}`")

                # Aquí no se selecciona columna, el modulo_1st.py ya tiene las columnas predefinidas
                # Si quisieras permitir la selección, tendrías que ajustar modulo_1st.py para aceptar una lista de columnas
                
                if st.button("🧠 Ejecutar Análisis Cuantitativo Básico", key="run_basic_analysis"):
                    with st.spinner("Analizando datos y generando informe..."):
                        all_results, word_doc_bytes, status = m1.analyze_excel_data_streamlit(df_basic)
                        
                        if status == "Success":
                            st.success("Análisis completado exitosamente.")
                            st.subheader("Resultados del Análisis:")
                            # Mostrar resultados en Streamlit
                            for item_type, title, content in all_results:
                                if item_type == 'text':
                                    # Usar st.markdown para renderizar el formato del texto (ej. negritas)
                                    st.markdown(content) 
                                elif item_type == 'image_bytes':
                                    st.image(content.getvalue(), caption=title, use_column_width=True)
                            
                            if word_doc_bytes:
                                st.download_button(
                                    label="💾 Descargar Informe Word Básico",
                                    data=word_doc_bytes,
                                    file_name=f"Informe_Analisis_Cuantitativo_Basico_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx",
                                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                    key="download_word_basic"
                                )
                            else:
                                st.warning("No se pudo generar el informe Word.")
                        elif status == "NoData":
                            st.warning("No se encontraron datos válidos para analizar.")
                        elif status == "NoValidColumns":
                            st.warning("No se encontraron columnas válidas para el análisis en el archivo subido según las especificaciones del módulo.")
                        else:
                            st.error("Ocurrió un error inesperado durante el análisis.")

            except Exception as e:
                st.error(f"Error al cargar o procesar el archivo Excel: {e}")
        else:
            st.info("Por favor, sube un archivo Excel para realizar el análisis cuantitativo básico.")

    # --- Sección: Análisis Cuantitativo y Comparativo ---
    elif app_choice == "Análisis Cuantitativo y Comparativo":
        st.header("📊 Análisis Cuantitativo y Comparativo")
        st.write("Realiza análisis cuantitativos avanzados, permitiendo comparaciones por múltiples criterios y aplicando filtros.")

        uploaded_file_comp_analysis = st.file_uploader(
            "Sube tu archivo Excel (.xlsx) para análisis comparativo",
            type=["xlsx"],
            key="comp_analysis_uploader"
        )

        if uploaded_file_comp_analysis:
            try:
                df_comp = pd.read_excel(uploaded_file_comp_analysis)
                st.subheader("Vista Previa de los Datos:")
                st.dataframe(df_comp.head())
                st.write(f"Columnas detectadas: `{', '.join(df_comp.columns.tolist())}`")

                # --- Opciones de Agrupación y Filtro ---
                st.subheader("Opciones de Agrupación y Filtrado:")
                
                # Criterios de agrupación principal
                st.markdown("##### Agrupar Comparación Principal Por:")
                group_criteria_options = ["Unidad", "Año de Residencia", "Sexo", "Especialidad"]
                selected_group_criteria = st.multiselect(
                    "Selecciona uno o más criterios para agrupar los datos:",
                    options=group_criteria_options,
                    key="main_group_criteria"
                )

                # Diccionarios para almacenar selecciones de valores específicos por grupo
                selected_ud_values = []
                selected_years_values = []
                selected_sexes_values = []
                selected_specialties_values = []

                # Mostrar selectores de valores específicos solo si el criterio principal está seleccionado
                if "Unidad" in selected_group_criteria:
                    # Obtener los números de UD presentes en los datos
                    ud_col_name = m3._get_pandas_column_name(df_comp, 'A')
                    if ud_col_name and ud_col_name in df_comp.columns:
                        # Convertir a numérico y luego a string para mapeo, manejar errores
                        unique_uds_numbers = pd.to_numeric(df_comp[ud_col_name], errors='coerce').dropna().astype(int).unique().tolist()
                        # Mapear números a nombres para mostrar en el multiselect
                        ud_display_options = [f"{m3.UD_MAPPING.get(num, str(num))} (UD {num})" for num in sorted(unique_uds_numbers)]
                        selected_ud_display = st.multiselect(
                            "Selecciona Unidades (UDs) específicas para comparar (si no seleccionas, se usarán todas las presentes):",
                            options=ud_display_options,
                            key="select_uds"
                        )
                        # Convertir de vuelta a números para pasarlos a la función
                        selected_ud_values = [int(s.split('(UD ')[1][:-1]) for s in selected_ud_display]
                    else:
                        st.warning("Columna 'Unidad (A)' no encontrada para seleccionar UDs.")

                if "Año de Residencia" in selected_group_criteria:
                    year_col_name = m3._get_pandas_column_name(df_comp, 'B')
                    if year_col_name and year_col_name in df_comp.columns:
                        unique_years = df_comp[year_col_name].dropna().astype(str).unique().tolist()
                        selected_years_values = st.multiselect(
                            "Selecciona Años de Residencia específicos para comparar (si no seleccionas, se usarán todos):",
                            options=sorted(unique_years),
                            key="select_years"
                        )
                    else:
                        st.warning("Columna 'Año de Residencia (B)' no encontrada para seleccionar años.")

                if "Sexo" in selected_group_criteria:
                    sex_col_name = m3._get_pandas_column_name(df_comp, 'C')
                    if sex_col_name and sex_col_name in df_comp.columns:
                        unique_sexes = df_comp[sex_col_name].dropna().astype(str).unique().tolist()
                        selected_sexes_values = st.multiselect(
                            "Selecciona Sexos específicos para comparar (si no seleccionas, se usarán todos):",
                            options=sorted(unique_sexes),
                            key="select_sexes"
                        )
                    else:
                        st.warning("Columna 'Sexo (C)' no encontrada para seleccionar sexos.")

                if "Especialidad" in selected_group_criteria:
                    specialty_col_name = m3._get_pandas_column_name(df_comp, 'D')
                    if specialty_col_name and specialty_col_name in df_comp.columns:
                        unique_specialties = df_comp[specialty_col_name].dropna().astype(str).unique().tolist()
                        selected_specialties_values = st.multiselect(
                            "Selecciona Especialidades específicas para comparar (si no seleccionas, se usarán todas):",
                            options=sorted(unique_specialties),
                            key="select_specialties"
                        )
                    else:
                        st.warning("Columna 'Especialidad (D)' no encontrada para seleccionar especialidades.")

                st.markdown("##### Filtros Secundarios (Opcional):")
                col_filter_year, col_filter_sex, col_filter_specialty = st.columns(3)
                
                year_filter_options = ["Todos"]
                if m3._get_pandas_column_name(df_comp, 'B') in df_comp.columns:
                    year_filter_options.extend(sorted(df_comp[m3._get_pandas_column_name(df_comp, 'B')].dropna().astype(str).unique().tolist()))
                with col_filter_year:
                    year_filter_val = st.selectbox("Filtrar por Año de Residencia:", year_filter_options, key="filter_year")
                
                sex_filter_options = ["Todos"]
                if m3._get_pandas_column_name(df_comp, 'C') in df_comp.columns:
                    sex_filter_options.extend(sorted(df_comp[m3._get_pandas_column_name(df_comp, 'C')].dropna().astype(str).unique().tolist()))
                with col_filter_sex:
                    sex_filter_val = st.selectbox("Filtrar por Sexo:", sex_filter_options, key="filter_sex")
                
                specialty_filter_options = ["Todos"]
                if m3._get_pandas_column_name(df_comp, 'D') in df_comp.columns:
                    specialty_filter_options.extend(sorted(df_comp[m3._get_pandas_column_name(df_comp, 'D')].dropna().astype(str).unique().tolist()))
                with col_filter_specialty:
                    specialty_filter_val = st.selectbox("Filtrar por Especialidad:", specialty_filter_options, key="filter_specialty")

                filters_dict = {
                    'year': year_filter_val,
                    'sex': sex_filter_val,
                    'specialty': specialty_filter_val
                }

                # --- Selección de Columnas para Comparar ---
                st.subheader("Seleccionar Columnas a Comparar:")
                # Excluir las columnas de agrupación y filtro de la selección de preguntas
                cols_to_exclude = [
                    m3._get_pandas_column_name(df_comp, 'A'), # UD
                    m3._get_pandas_column_name(df_comp, 'B'), # Año
                    m3._get_pandas_column_name(df_comp, 'C'), # Sexo
                    m3._get_pandas_column_name(df_comp, 'D')  # Especialidad
                ]
                available_cols_for_comparison = [col for col in df_comp.columns if col not in cols_to_exclude]

                selected_cols_for_comparison = st.multiselect(
                    "Selecciona las columnas (preguntas) que deseas analizar comparativamente:",
                    options=available_cols_for_comparison,
                    key="columns_to_compare"
                )

                if st.button("🔄 Ejecutar Análisis Comparativo", key="run_comp_analysis"):
                    if not selected_group_criteria:
                        st.warning("Por favor, selecciona al menos un criterio de agrupación principal.")
                    elif not selected_cols_for_comparison:
                        st.warning("Por favor, selecciona al menos una columna de preguntas para comparar.")
                    else:
                        with st.spinner("Ejecutando análisis comparativo y generando informe..."):
                            all_results, word_doc_bytes, status = m3.run_comparison_analysis_streamlit(
                                df_input=df_comp,
                                selected_group_criteria_raw=selected_group_criteria,
                                selected_ud_numbers=selected_ud_values,
                                selected_years=selected_years_values,
                                selected_sexes=selected_sexes_values,
                                selected_specialties=selected_specialties_values,
                                filters=filters_dict,
                                selected_columns_for_comparison=selected_cols_for_comparison
                            )

                            if status == "Success":
                                st.success("Análisis comparativo completado exitosamente.")
                                st.subheader("Resultados del Análisis Comparativo:")
                                for item_type, title, content in all_results:
                                    if item_type == 'text':
                                        st.markdown(content)
                                    elif item_type == 'image_bytes':
                                        st.image(content.getvalue(), caption=title, use_column_width=True)
                                
                                if word_doc_bytes:
                                    st.download_button(
                                        label="💾 Descargar Informe Word Comparativo",
                                        data=word_doc_bytes,
                                        file_name=f"Informe_Analisis_Comparativo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx",
                                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                        key="download_word_comp"
                                    )
                                else:
                                    st.warning("No se pudo generar el informe Word.")
                            elif status == "NoData":
                                st.warning("No se encontraron datos válidos para analizar.")
                            elif status == "NoColumnsSelected":
                                st.warning("No se seleccionaron columnas para comparar.")
                            elif status == "NoDataAfterFilter":
                                st.warning("No hay datos que coincidan con los filtros y selecciones de grupo. Ajusta tus criterios.")
                            elif status == "NoGroupingCriteria":
                                st.warning("No se pudo iniciar el análisis comparativo. Verifica tus selecciones de criterios de agrupación y que las columnas correspondientes existan.")
                            else:
                                st.error("Ocurrió un error inesperado durante el análisis comparativo.")

            except Exception as e:
                st.error(f"Error al cargar o procesar el archivo Excel: {e}")
        else:
            st.info("Por favor, sube un archivo Excel para realizar el análisis comparativo.")

    # --- Sección: Análisis Cualitativo (PLN + IA) ---
    elif app_choice == "Análisis Cualitativo (PLN + IA)":
        st.header("🗣️ Análisis Cualitativo (PLN + IA)")
        st.write("Realiza análisis de texto avanzado (frecuencia, sentimiento, modelado de temas) e interactúa con modelos de Inteligencia Artificial.")

        uploaded_file_qual_analysis = st.file_uploader(
            "Sube tu archivo Excel (.xlsx) con columnas de texto para análisis",
            type=["xlsx"],
            key="qual_analysis_uploader"
        )

        if uploaded_file_qual_analysis:
            try:
                df_qual = pd.read_excel(uploaded_file_qual_analysis)
                st.subheader("Vista Previa de los Datos:")
                st.dataframe(df_qual.head())
                st.write(f"Columnas detectadas: `{', '.join(df_qual.columns.tolist())}`")

                text_columns_options = [col for col in df_qual.columns if df_qual[col].dtype == 'object']
                selected_text_columns = st.multiselect(
                    "Selecciona las columnas de texto para analizar:",
                    options=text_columns_options,
                    key="qual_text_columns"
                )

                st.subheader("Opciones de Análisis de PLN:")
                col_pln_freq, col_pln_sent, col_pln_topic = st.columns(3)
                with col_pln_freq:
                    run_frequency = st.checkbox("Análisis de Frecuencia", value=True, key="run_freq")
                with col_pln_sent:
                    run_sentiment = st.checkbox("Análisis de Sentimiento", value=True, key="run_sentiment")
                with col_pln_topic:
                    run_topic_modeling = st.checkbox("Modelado de Temas", value=False, key="run_topic_modeling")

                analysis_options = {
                    'frequency': run_frequency,
                    'sentiment': run_sentiment,
                    'topic_modeling': run_topic_modeling
                }

                if run_topic_modeling:
                    num_topics = st.slider("Número de Temas (para Modelado de Temas):", min_value=2, max_value=10, value=5, key="num_topics")
                    topic_method = st.selectbox("Método de Modelado de Temas:", ["NMF", "LDA"], key="topic_method")
                    analysis_options['num_topics'] = num_topics
                    analysis_options['topic_method'] = topic_method

                st.subheader("Interacción con Modelos de IA:")
                use_ai = st.checkbox("Activar Interacción con IA", key="use_ai_checkbox")
                ai_model_choice = None
                ai_prompt = None

                if use_ai:
                    ai_model_options = ["Selecciona un modelo", "Google Gemini", "OpenAI GPT", "Anthropic Claude"] # Añadir "Meta Llama 3" si lo implementas
                    ai_model_choice = st.selectbox("Selecciona un Modelo de IA:", ai_model_options, key="ai_model_choice")
                    
                    # Opción para usar el creador de prompts o introducir uno manualmente
                    prompt_source = st.radio(
                        "¿Cómo quieres obtener el prompt para la IA?",
                        ("Introducir manualmente", "Usar Creador de Prompts"),
                        key="prompt_source_radio"
                    )

                    if prompt_source == "Introducir manualmente":
                        ai_prompt = st.text_area("Introduce tu prompt para la IA aquí:", height=150, key="manual_ai_prompt")
                    else: # Usar Creador de Prompts
                        st.markdown("---")
                        st.subheader("Crea tu Prompt con el Asistente:")
                        # Lógica del creador de prompts integrada aquí
                        verbs = prompt_creator.get_all_verbs()
                        role_categories = prompt_creator.get_role_categories()

                        col_verb, col_cat = st.columns(2)
                        with col_verb:
                            selected_verb = st.selectbox("Verbo de Acción:", verbs, key="prompt_verb")
                        with col_cat:
                            selected_category = st.selectbox("Categoría de Rol:", role_categories, key="prompt_category")
                        
                        roles = prompt_creator.get_roles_by_category(selected_category)
                        selected_role = st.selectbox("Rol Específico:", roles, key="prompt_role")
                        
                        additional_context = st.text_area("Contexto/Instrucciones Adicionales:", key="prompt_context")
                        
                        generated_prompt_display = prompt_creator.generate_ai_prompt(selected_verb, selected_role, additional_context)
                        st.text_area("Prompt Generado (automáticamente copiado para la IA):", value=generated_prompt_display, height=100, key="generated_ai_prompt_display")
                        ai_prompt = generated_prompt_display # Asignar el prompt generado para usarlo en la IA

                        st.markdown("---")


                if st.button("🧠 Ejecutar Análisis Cualitativo", key="run_qual_analysis"):
                    if not selected_text_columns:
                        st.warning("Por favor, selecciona al menos una columna de texto para el análisis.")
                    elif use_ai and (ai_model_choice == "Selecciona un modelo" or not ai_prompt or ai_prompt == "Tu prompt aparecerá aquí."):
                        st.warning("Si activas la IA, debes seleccionar un modelo y proporcionar un prompt válido.")
                    else:
                        with st.spinner("Analizando texto y consultando IA (si aplica)..."):
                            all_results, word_doc_bytes, status = m2.run_qualitative_analysis_streamlit(
                                df_input=df_qual,
                                text_columns=selected_text_columns,
                                analysis_options=analysis_options,
                                ai_prompt=ai_prompt,
                                ai_model_choice=ai_model_choice if use_ai else None
                            )

                            if status == "Success":
                                st.success("Análisis cualitativo completado exitosamente.")
                                st.subheader("Resultados del Análisis Cualitativo:")
                                for item_type, title, content in all_results:
                                    if item_type == 'text':
                                        st.markdown(f"### {title}") # Usar markdown para títulos
                                        st.markdown(content)
                                    elif item_type == 'image_bytes':
                                        st.image(content.getvalue(), caption=title, use_column_width=True)
                                
                                if word_doc_bytes:
                                    st.download_button(
                                        label="💾 Descargar Informe Word Cualitativo",
                                        data=word_doc_bytes,
                                        file_name=f"Informe_Analisis_Cualitativo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx",
                                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                        key="download_word_qual"
                                    )
                                else:
                                    st.warning("No se pudo generar el informe Word.")
                            elif status == "NoData":
                                st.warning("No se encontraron datos válidos para analizar.")
                            elif status == "NoTextColumns":
                                st.warning("No se seleccionaron columnas de texto para el análisis.")
                            else:
                                st.error("Ocurrió un error inesperado durante el análisis cualitativo.")

            except Exception as e:
                st.error(f"Error al cargar o procesar el archivo Excel: {e}")
        else:
            st.info("Por favor, sube un archivo Excel para realizar el análisis cualitativo.")

    # --- Sección: Creador de Prompts IA (como herramienta independiente) ---
    elif app_choice == "Creador de Prompts IA":
        st.header("📝 Creador de Prompts para IA")
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

        st.info("Para copiar el prompt, selecciona el texto en el cuadro de arriba y usa `Ctrl+C` (o `Cmd+C` en Mac).")


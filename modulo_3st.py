# modulo_3st.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from docx import Document
from docx.shared import Inches
import io
import numpy as np
import collections
from datetime import datetime
import streamlit as st # Importar Streamlit

# Configuración global de gráficos
plt.rcParams["figure.figsize"] = (10, 6) # Ajustado para Streamlit
sns.set_style("whitegrid")

# Mapeo de Unidades (UDs)
UD_MAPPING = {
    1: "HUGCDRN", 2: "CHUIMI", 3: "CHUC", 4: "HUNSC", 5: "GSSLZ",
    6: "AFyC GC", 7: "AFyC TFNorte", 8: "AFyC TFSur", 9: "AFyC FV",
    10: "AFyC LP", 11: "SM GC", 12: "SM TF", 13: "ObsGin CHUIMI",
    14: "ENF ObsGin", 15: "MPySP", 16: "SL", 17: "PED CHUIMI", 18: "MED Legal"
}

# --- Funciones Auxiliares (adaptadas para Streamlit) ---

def _log_message_streamlit(message, level="info"):
    """
    Función auxiliar para mostrar mensajes en la interfaz de Streamlit.
    """
    if level == "info":
        st.info(message)
    elif level == "warning":
        st.warning(message)
    elif level == "error":
        st.error(message)
    elif level == "success":
        st.success(message)
    else:
        st.write(message) # Default

def get_excel_column_letter(index):
    """Convierte un índice de columna (0-based) a su letra de Excel (A, B, C...)."""
    if index < 0: return None
    result = ""
    while index >= 0:
        result = chr(65 + (index % 26)) + result
        index = (index // 26) - 1
    return result

def _get_pandas_column_name(df, excel_letter):
    """Obtiene el nombre de la columna de Pandas a partir de la letra de Excel."""
    # Esto es una asunción basada en el mapeo inicial de A, B, C, D
    # Si la estructura de tu Excel es diferente, esto necesitará ser más robusto.
    if excel_letter == 'A':
        return df.columns[0] if len(df.columns) > 0 else None
    elif excel_letter == 'B':
        return df.columns[1] if len(df.columns) > 1 else None
    elif excel_letter == 'C':
        return df.columns[2] if len(df.columns) > 2 else None
    elif excel_letter == 'D':
        return df.columns[3] if len(df.columns) > 3 else None
    return None # No se encontró la columna de Excel mapeada


def _apply_filters(df, filters, col_A_name, col_B_name, col_C_name, col_D_name):
    """Aplica los filtros secundarios al DataFrame."""
    filtered_df = df.copy()

    year_filter = filters.get('year', "Todos")
    sex_filter = filters.get('sex', "Todos")
    specialty_filter = filters.get('specialty', "Todos")

    if year_filter != "Todos" and col_B_name in filtered_df.columns:
        filtered_df = filtered_df[filtered_df[col_B_name].astype(str) == str(year_filter)]
    if sex_filter != "Todos" and col_C_name in filtered_df.columns:
        filtered_df = filtered_df[filtered_df[col_C_name].astype(str).str.lower() == str(sex_filter).lower()]
    if specialty_filter != "Todos" and col_D_name in filtered_df.columns:
        filtered_df = filtered_df[filtered_df[col_D_name].astype(str).str.lower() == str(specialty_filter).lower()]

    return filtered_df

def _generate_comparison_chart(data, col_name, group_criterion_name, chart_type='bar'):
    """Genera un gráfico comparativo (barras o pastel) para una columna y criterio de agrupación."""
    fig, ax = plt.subplots(figsize=(12, 7))

    if chart_type == 'bar':
        # Asegurarse de que el índice es numérico o de string para el eje x
        if pd.api.types.is_numeric_dtype(data.index):
            data.plot(kind='bar', ax=ax, width=0.8, color=plt.cm.viridis(np.linspace(0, 1, len(data.columns))))
        else: # Asumimos categórico (e.g., UDs, Sexo, Especialidad)
            data.plot(kind='bar', ax=ax, width=0.8, color=plt.cm.viridis(np.linspace(0, 1, len(data.columns))))
            
        ax.set_title(f"Comparación de Frecuencia para '{col_name}' por {group_criterion_name}", fontsize=14)
        ax.set_xlabel(group_criterion_name, fontsize=12)
        ax.set_ylabel("Frecuencia", fontsize=12)
        ax.tick_params(axis='x', rotation=45, ha='right')
        ax.legend(title="Categorías de Respuesta", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Ajusta el espacio para la leyenda

        # Añadir porcentajes sobre las barras
        for container in ax.containers:
            for i, rect in enumerate(container.patches):
                total_for_group = data.iloc[i].sum()
                height = rect.get_height()
                if total_for_group > 0:
                    percentage = (height / total_for_group) * 100
                    ax.text(rect.get_x() + rect.get_width() / 2, height, f'{percentage:.1f}%', ha='center', va='bottom', fontsize=8, color='black')

    elif chart_type == 'pie':
        # Para gráficos de pastel, cada columna de 'data' es un conjunto de categorías a comparar.
        # Generalmente, el pastel se hace para UNA SERIE, no múltiples series como en un bar plot comparativo.
        # Para comparación de pastel, se debería generar un pastel por cada grupo.
        # Simplificamos a un bar chart para comparación de categorías entre grupos.
        _log_message_streamlit("Los gráficos de pastel no son adecuados para la comparación entre múltiples grupos. Se generará un gráfico de barras.", "warning")
        return _generate_comparison_chart(data, col_name, group_criterion_name, chart_type='bar')

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=200)
    buf.seek(0)
    plt.close(fig) # Liberar memoria
    return buf


def _generate_numeric_comparison_chart(data, col_name, group_criterion_name, metric='mean'):
    """Genera un gráfico de barras para la comparación de métricas numéricas."""
    fig, ax = plt.subplots(figsize=(10, 6))

    if metric == 'mean':
        title_metric = "Promedio"
    elif metric == 'median':
        title_metric = "Mediana"
    elif metric == 'std':
        title_metric = "Desviación Estándar"
    else:
        title_metric = "Valor"

    data.plot(kind='bar', ax=ax, color=plt.cm.Paired(np.arange(len(data))), edgecolor='black')
    ax.set_title(f"Comparación de {title_metric} de '{col_name}' por {group_criterion_name}", fontsize=14)
    ax.set_xlabel(group_criterion_name, fontsize=12)
    ax.set_ylabel(title_metric, fontsize=12)
    ax.tick_params(axis='x', rotation=45, ha='right')
    plt.tight_layout()

    # Añadir valores sobre las barras
    for container in ax.containers:
        for rect in container.patches:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom', fontsize=9, color='black')

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=200)
    buf.seek(0)
    plt.close(fig) # Liberar memoria
    return buf

def export_comparison_to_word(results_sequence, comparison_summary):
    """
    Genera un documento Word con los resultados del análisis comparativo.
    Recibe una secuencia de tuplas: ('text', title, content) o ('image_bytes', title, bytes_io_obj).
    Devuelve un objeto BytesIO que contiene el documento Word.
    """
    if not results_sequence:
        return None

    doc = Document()
    doc.add_heading("📊 Informe de Análisis Cuantitativo y Comparativo", 0)
    doc.add_paragraph("Este informe presenta un análisis comparativo detallado de las preguntas seleccionadas, agrupadas por los criterios elegidos y aplicando los filtros correspondientes.\n")
    
    # Resumen de las selecciones
    doc.add_heading("Resumen de las Opciones de Análisis", level=1)
    doc.add_paragraph(f"Criterio(s) de Agrupación Principal: {comparison_summary.get('main_criterion', 'N/A')}")
    doc.add_paragraph(f"Grupos Seleccionados: {', '.join(comparison_summary.get('selected_groups', ['N/A']))}")
    filters_text = ", ".join([f"{k}: {v}" for k, v in comparison_summary.get('filters_applied', {}).items() if v != "Todos"])
    doc.add_paragraph(f"Filtros Aplicados: {filters_text if filters_text else 'Ninguno'}")
    doc.add_paragraph("\n")


    for item_type, title, content in results_sequence:
        if item_type == 'text':
            doc.add_heading(title, level=2) # Usar título como subencabezado
            doc.add_paragraph(content)
        elif item_type == 'image_bytes':
            try:
                content.seek(0) # Resetear el puntero del BytesIO antes de usarlo
                doc.add_picture(content, width=Inches(6.5)) # Ancho un poco mayor
            except Exception as e:
                doc.add_paragraph(f"(Error al insertar imagen {title}: {e})")
    
    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer

# --- Función Principal para Streamlit ---

def run_comparison_analysis_streamlit(
    df_input,
    selected_group_criteria_raw, # Lista de "UD", "Año", "Sexo", "Especialidad"
    selected_ud_numbers,         # Lista de números de UD seleccionados
    selected_years,              # Lista de años seleccionados
    selected_sexes,              # Lista de sexos seleccionados
    selected_specialties,        # Lista de especialidades seleccionadas
    filters,                     # Diccionario {'year': val, 'sex': val, 'specialty': val}
    selected_columns_for_comparison # Lista de nombres de columnas de Pandas
):
    """
    Función principal para ejecutar el análisis comparativo en Streamlit.
    """
    _log_message_streamlit("⏳ Iniciando análisis comparativo...", "info")

    if df_input is None or df_input.empty:
        _log_message_streamlit("No se han cargado datos válidos para el análisis.", "warning")
        return [], None, "NoData"

    if not selected_columns_for_comparison:
        _log_message_streamlit("Por favor, selecciona al menos una columna de preguntas para comparar.", "warning")
        return [], None, "NoColumnsSelected"

    # Mapear letras de Excel a nombres de columnas de Pandas (asumiendo orden)
    # Este mapeo es crítico y debe reflejar el archivo Excel real.
    # Si las columnas 'Unidad', 'Año de Residencia', 'Sexo', 'Especialidad' no son A, B, C, D
    # esto necesitará ser ajustado.
    col_A_name = _get_pandas_column_name(df_input, 'A') # Unidad
    col_B_name = _get_pandas_column_name(df_input, 'B') # Año de Residencia
    col_C_name = _get_pandas_column_name(df_input, 'C') # Sexo
    col_D_name = _get_pandas_column_name(df_input, 'D') # Especialidad

    # Validar que las columnas necesarias existan
    required_cols_map = {
        "Unidad": col_A_name,
        "Año de Residencia": col_B_name,
        "Sexo": col_C_name,
        "Especialidad": col_D_name
    }
    
    # Filtrar el DataFrame
    filtered_df = _apply_filters(df_input, filters, col_A_name, col_B_name, col_C_name, col_D_name)

    if filtered_df.empty:
        _log_message_streamlit("El DataFrame filtrado no contiene datos. Ajusta tus filtros.", "warning")
        return [], None, "NoDataAfterFilter"
    
    analysis_results_sequence = []
    comparison_summary_details = {
        "main_criterion": ", ".join(selected_group_criteria_raw),
        "selected_groups": [],
        "filters_applied": filters
    }

    # === Preparación de los criterios de agrupación ===
    group_criteria_mapping = {
        "Unidad": (col_A_name, selected_ud_numbers, UD_MAPPING),
        "Año de Residencia": (col_B_name, selected_years, None), # No hay mapeo especial para años
        "Sexo": (col_C_name, selected_sexes, None),
        "Especialidad": (col_D_name, selected_specialties, None)
    }

    actual_group_criteria_to_process = []
    
    for crit_display_name in selected_group_criteria_raw:
        col_pandas_name, selected_values, mapping = group_criteria_mapping.get(crit_display_name)
        if not col_pandas_name or col_pandas_name not in filtered_df.columns:
            _log_message_streamlit(f"La columna para el criterio '{crit_display_name}' no se encontró en los datos o no está mapeada correctamente.", "warning")
            continue
        
        # Filtrar el DataFrame según los valores seleccionados para este criterio de agrupación
        # Importante: para UD, se filtra por el número, no por el nombre de la UD
        if crit_display_name == "Unidad":
            if selected_values:
                # Mapear los nombres de UD seleccionados de nuevo a números para el filtrado
                selected_ud_names = [UD_MAPPING[num] for num in selected_values if num in UD_MAPPING]
                # Ahora filtrar la columna de Unidad que contiene los nombres de UD
                df_to_analyze_group = filtered_df[filtered_df[col_pandas_name].isin(selected_ud_names)].copy()
                # Asegurarse de que el criterio de agrupación se mantenga como el nombre de la UD para la visualización
                group_col_for_plotting = col_pandas_name
                selected_group_names = selected_ud_names
            else:
                df_to_analyze_group = filtered_df.copy()
                group_col_for_plotting = col_pandas_name
                selected_group_names = [name for num, name in UD_MAPPING.items() if num in filtered_df[col_pandas_name].unique()] # Todos los nombres de UD presentes
        else:
            if selected_values:
                df_to_analyze_group = filtered_df[filtered_df[col_pandas_name].astype(str).isin([str(v) for v in selected_values])].copy()
                group_col_for_plotting = col_pandas_name
                selected_group_names = [str(v) for v in selected_values]
            else:
                df_to_analyze_group = filtered_df.copy()
                group_col_for_plotting = col_pandas_name
                selected_group_names = df_to_analyze_group[col_pandas_name].astype(str).unique().tolist() # Todos los valores únicos en la columna de grupo

        if df_to_analyze_group.empty:
            _log_message_streamlit(f"No hay datos para comparar para el criterio '{crit_display_name}' con los grupos seleccionados.", "warning")
            continue

        actual_group_criteria_to_process.append({
            "display_name": crit_display_name,
            "pandas_col_name": col_pandas_name,
            "df_filtered_for_group": df_to_analyze_group,
            "selected_group_names": selected_group_names
        })
        # Actualizar el resumen para el informe
        comparison_summary_details["selected_groups"].extend(selected_group_names)


    if not actual_group_criteria_to_process:
        _log_message_streamlit("No se pudo iniciar el análisis comparativo. Verifica tus selecciones de criterios de agrupación y que las columnas correspondientes existan.", "warning")
        return [], None, "NoGroupingCriteria"
    
    # Asegurarse de que los selected_groups sean únicos para el resumen
    comparison_summary_details["selected_groups"] = list(set(comparison_summary_details["selected_groups"]))

    # === Ejecutar Comparación ===
    for col_question in selected_columns_for_comparison:
        if col_question not in filtered_df.columns:
            _log_message_streamlit(f"La columna de pregunta '{col_question}' no se encontró en el DataFrame filtrado. Saltando.", "warning")
            continue
        
        analysis_results_sequence.append(('text', f"Análisis Comparativo para: {col_question}", ""))

        for group_crit_info in actual_group_criteria_to_process:
            crit_display_name = group_crit_info["display_name"]
            group_pandas_col_name = group_crit_info["pandas_col_name"]
            df_for_group_analysis = group_crit_info["df_filtered_for_group"]
            selected_group_names_for_this_crit = group_crit_info["selected_group_names"] # Ya filtrado por los valores seleccionados

            # Asegurarse de que la columna existe en el DF para este grupo
            if group_pandas_col_name not in df_for_group_analysis.columns:
                _log_message_streamlit(f"La columna de agrupación '{group_pandas_col_name}' no está presente en el DataFrame para el criterio '{crit_display_name}'.", "warning")
                continue

            # Convertir la columna de pregunta a numérica si es posible para estadísticas
            is_numeric_question = pd.api.types.is_numeric_dtype(df_for_group_analysis[col_question])

            if is_numeric_question:
                # Análisis para columnas numéricas: promedio, mediana, etc.
                grouped_stats = df_for_group_analysis.groupby(group_pandas_col_name)[col_question].agg(['mean', 'median', 'std']).reset_index()
                grouped_stats.rename(columns={'mean': 'Promedio', 'median': 'Mediana', 'std': 'Desviación Estándar'}, inplace=True)
                
                text_result = f"\n📊 Estadísticas numéricas de '{col_question}' agrupadas por '{crit_display_name}':\n"
                text_result += grouped_stats.to_string(index=False) + "\n"
                analysis_results_sequence.append(('text', f"Estadísticas por {crit_display_name} para {col_question}", text_result))
                
                # Gráficos para numéricas
                # Gráfico de Promedio
                if 'Promedio' in grouped_stats.columns:
                    plot_data = grouped_stats.set_index(group_pandas_col_name)['Promedio']
                    img_buf = _generate_numeric_comparison_chart(plot_data, col_question, crit_display_name, metric='mean')
                    analysis_results_sequence.append(('image_bytes', f"Promedio de {col_question} por {crit_display_name}", img_buf))

                # Gráfico de Mediana
                if 'Mediana' in grouped_stats.columns:
                    plot_data = grouped_stats.set_index(group_pandas_col_name)['Mediana']
                    img_buf = _generate_numeric_comparison_chart(plot_data, col_question, crit_display_name, metric='median')
                    analysis_results_sequence.append(('image_bytes', f"Mediana de {col_question} por {crit_display_name}", img_buf))

            else:
                # Análisis para columnas categóricas: conteo de frecuencias
                # Filtrar el DataFrame antes de contar para incluir solo los grupos seleccionados si aplica
                if crit_display_name == "Unidad" and selected_ud_numbers:
                    df_filtered_by_group = df_for_group_analysis[df_for_group_analysis[group_pandas_col_name].isin(selected_group_names_for_this_crit)]
                elif selected_group_names_for_this_crit: # Para Año, Sexo, Especialidad, se filtra por los nombres directamente
                     df_filtered_by_group = df_for_group_analysis[df_for_group_analysis[group_pandas_col_name].astype(str).isin([str(g) for g in selected_group_names_for_this_crit])]
                else: # Si no hay grupos seleccionados para este criterio, usar todo el df_for_group_analysis
                    df_filtered_by_group = df_for_group_analysis

                # Realizar el crosstab para obtener frecuencias de la pregunta por grupo
                contingency_table = pd.crosstab(
                    df_filtered_by_group[group_pandas_col_name],
                    df_filtered_by_group[col_question]
                )
                
                # Calcular porcentajes
                contingency_table_percent = contingency_table.apply(lambda r: r/r.sum() * 100, axis=1)

                text_result = f"\n📈 Distribución de '{col_question}' por '{crit_display_name}' (Frecuencias):\n"
                text_result += contingency_table.to_string() + "\n\n"
                text_result += f"Distribución de '{col_question}' por '{crit_display_name}' (Porcentajes):\n"
                text_result += contingency_table_percent.to_string(float_format="%.2f") + "%\n"
                analysis_results_sequence.append(('text', f"Distribución por {crit_display_name} para {col_question}", text_result))
                
                # Gráfico de barras apiladas o agrupadas
                if not contingency_table_percent.empty:
                    img_buf = _generate_comparison_chart(contingency_table_percent, col_question, crit_display_name, chart_type='bar')
                    analysis_results_sequence.append(('image_bytes', f"Comparación de {col_question} por {crit_display_name}", img_buf))
            
            analysis_results_sequence.append(('text', "", "-" * 80 + "\n")) # Separador

    word_document_bytes = export_comparison_to_word(analysis_results_sequence, comparison_summary_details)
    _log_message_streamlit("✅ Análisis comparativo completado.", "success")
    return analysis_results_sequence, word_document_bytes, "Success"
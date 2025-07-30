# modulo_1st.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from docx import Document
from docx.shared import Inches
import os
import numpy as np
import io # Para manejar archivos en memoria
from datetime import datetime

# Configuración global de gráficos
plt.rcParams["figure.figsize"] = (8, 5) # Tamaño fijo para todas las figuras (ancho, alto en pulgadas)
sns.set_style("whitegrid") # Estilo de fondo para gráficos

# No necesitamos tkinter, ttk, filedialog, messagebox, Image, ImageTk, ImageOps aquí
# st.pyplot ya maneja las figuras de matplotlib

# Función auxiliar para mostrar mensajes en la interfaz de Streamlit
def _log_message_streamlit(message, level="info"):
    if level == "info":
        st.info(message)
    elif level == "warning":
        st.warning(message)
    elif level == "error":
        st.error(message)
    elif level == "success":
        st.success(message)
    else:
        st.write(message) # Default para cualquier otro nivel

def get_excel_column_letter(index):
    """Convierte un índice de columna (0-based) a su letra de Excel (A, B, C...)."""
    if index < 0: return None
    result = ""
    while index >= 0:
        result = chr(65 + (index % 26)) + result
        index = (index // 26) - 1
    return result

def generar_explicacion_analisis(col_title_display, data, analysis_type, graph_style):
    explanation = ""
    objective_interpretation = ""

    if analysis_type == 'numeric_continuous':
        serie = data
        mean_val = serie.mean()
        median_val = serie.median()
        std_val = serie.std()
        min_val = serie.min()
        max_val = serie.max()

        explanation += f"Interpretación de la distribución de {col_title_display}:\n"
        
        if std_val > 0:
            diff_ratio = (mean_val - median_val) / std_val
            if diff_ratio > 0.5:
                explanation += "La distribución está ligeramente sesgada a la derecha (positivamente), lo que sugiere la presencia de algunos valores altos. "
            elif diff_ratio < -0.5:
                explanation += "La distribución está ligeramente sesgada a la izquierda (negativamente), lo que indica la presencia de algunos valores bajos. "
            else:
                explanation += "La distribución parece ser relativamente simétrica, con la media y la mediana cercanas entre sí. "
        else:
            explanation += "Todos los valores en esta columna son idénticos, resultando en una desviación estándar de cero. "

        explanation += f"La desviación estándar de {std_val:.2f} indica la dispersión de los datos alrededor de la media. "
        explanation += f"Los valores de {col_title_display} oscilan entre {min_val:.2f} (mínimo) y {max_val:.2f} (máximo). "
        
        objective_interpretation = f"Este gráfico de barras vertical visualiza la concentración de los datos numéricos para '{col_title_display}'. Permite identificar los rangos de valores más frecuentes y la dispersión general, ayudando a entender la variabilidad en las respuestas de esta pregunta."

    elif analysis_type == 'categorical_standard':
        serie = data
        value_counts = serie.value_counts(normalize=True) * 100
        explanation += f"Distribución de {col_title_display}:\n"

        if not value_counts.empty:
            top_category = value_counts.index[0]
            top_percentage = value_counts.iloc[0]
            explanation += f"La categoría más frecuente es '{top_category}' con un {top_percentage:.1f}% de ocurrencias. "
            if len(value_counts) > 1:
                second_category = value_counts.index[1]
                second_percentage = value_counts.iloc[1]
                explanation += f"Seguida por '{second_category}' con un {second_percentage:.1f}%. "
            explanation += "Esto indica una concentración de datos en las categorías principales. "
            
            if graph_style == 'pie':
                objective_interpretation = f"El gráfico circular (de pastel) para '{col_title_display}' muestra visualmente la proporción de cada categoría. La mayoría de los participantes se identifican con '{top_category}' ({top_percentage:.1f}%), lo que resalta una característica o preferencia predominante. Los porcentajes facilitan una comprensión directa de la distribución total."
            else: # Gráfico de barras (vertical u horizontal)
                objective_interpretation = f"El gráfico de barras para '{col_title_display}' ilustra la distribución de las respuestas entre las diferentes categorías. Es evidente que '{top_category}' es la opción más común, indicando una clara tendencia o preferencia. Los porcentajes sobre las barras facilitan la comparación de la frecuencia relativa de cada respuesta."
        else:
            objective_interpretation = "No se detectaron categorías distintivas en esta columna para una interpretación."
    
    elif analysis_type == 'phrase_frequency_by_response':
        response_counts = data # data es la Serie de value_counts()
        total_responses = response_counts.sum()
        explanation += f"Análisis de frecuencia de respuestas en {col_title_display}:\n"

        if total_responses > 0:
            most_common_response = response_counts.index[0]
            most_common_count = response_counts.iloc[0]
            explanation += f"La respuesta más frecuente es '{most_common_response}' apareciendo {most_common_count} veces, lo que representa un {(most_common_count/total_responses)*100:.2f}% del total de respuestas válidas en esta columna. "
            explanation += "Esto indica qué respuestas son las más comunes o predominantes."
            
            objective_interpretation = f"Este gráfico de barras horizontal muestra la distribución de las respuestas únicas para la columna '{col_title_display}'. Es útil para identificar rápidamente las categorías de respuestas más frecuentes y su proporción sobre el total, ofreciendo una visión clara de los patrones de respuesta a esta pregunta abierta."
        else:
            objective_interpretation = "No se encontraron respuestas válidas para analizar la frecuencia en esta columna."

    return explanation + "\n" + objective_interpretation + "\n"

def generar_grafico(col, data_to_plot_raw, graph_style, analysis_type):
    fig, ax = plt.subplots() 
    
    try:
        ax.set_title(col) # Título del gráfico

        if graph_style == 'pie':
            counts = data_to_plot_raw.value_counts()
            colors = sns.color_palette("Paired", n_colors=len(counts)) 
            
            wedges, texts, autotexts = ax.pie(counts, labels=counts.index.astype(str), autopct='%1.1f%%',
                                              startangle=90, colors=colors, wedgeprops=dict(edgecolor='black'))
            ax.axis('equal')  
            
            for autotext in autotexts:
                autotext.set_color('black') 
                autotext.set_fontsize(9)
        
        elif graph_style == 'bar_vertical':
            if analysis_type == 'numeric_continuous' and data_to_plot_raw.nunique() > 15 and pd.api.types.is_numeric_dtype(data_to_plot_raw):
                # Si es numérica y tiene muchos valores únicos, hacer un histograma
                colors = sns.color_palette("viridis", n_colors=1)
                sns.histplot(data_to_plot_raw, kde=True, ax=ax, color=colors[0], edgecolor="black")
                ax.set_ylabel("Frecuencia")
                ax.set_xlabel(col)
            else: # Categórica (B, o numéricas discretas con pocos valores únicos)
                counts = data_to_plot_raw.value_counts().sort_index()
                colors = sns.color_palette("viridis", n_colors=len(counts)) 
                bars = ax.bar(counts.index.astype(str), counts.values, color=colors, edgecolor="black")
                ax.set_ylabel("Frecuencia")
                ax.set_xlabel("") 
                for label in ax.get_xticklabels():
                    label.set_rotation(45)
                    label.set_horizontalalignment('right')
                total = counts.sum()
                for bar in bars:
                    height = bar.get_height()
                    percentage = (height / total) * 100
                    ax.text(bar.get_x() + bar.get_width() / 2, height,
                            f'{percentage:.1f}%', ha='center', va='bottom', fontsize=8)

        elif graph_style == 'bar_horizontal':
            if analysis_type == 'phrase_frequency_by_response':
                counts = data_to_plot_raw # Ya viene como conteos
            else: # analysis_type == 'categorical_standard'
                counts = data_to_plot_raw.value_counts() # Contar los valores

            # Limitar a las N más comunes si hay muchas categorías para las frases
            if analysis_type == 'phrase_frequency_by_response' and len(counts) > 20: 
                counts = counts.nlargest(20) # Tomar las 20 respuestas más comunes
            
            counts.index = counts.index.astype(str)

            colors = sns.color_palette("viridis", n_colors=len(counts)) 
            bars = ax.barh(counts.index, counts.values, color=colors, edgecolor="black")
            ax.set_xlabel("Frecuencia")
            ax.set_ylabel("") 
            
            total = counts.sum()
            for bar in bars:
                width = bar.get_width()
                percentage = (width / total) * 100 if total > 0 else 0
                ax.text(width, bar.get_y() + bar.get_height()/2,
                        f'{percentage:.1f}%', va='center', ha='left', fontsize=8)
            
            ax.invert_yaxis() # Para que la barra más frecuente quede arriba
            fig.tight_layout(pad=3.0) 

        return fig # Devolver la figura de matplotlib

    except Exception as e:
        _log_message_streamlit(f"Error al generar el gráfico para '{col}' ({graph_style}, {analysis_type}): {e}", "error")
        plt.close(fig) # Asegurarse de cerrar la figura si hay un error
        return None # Devolver None si hay un error

def export_to_word_streamlit(results_sequence):
    """
    Genera un documento Word con los resultados del análisis.
    Recibe una secuencia de tuplas: ('text', col_name, content) o ('image_bytes', col_name, bytes_io_obj).
    Devuelve un objeto BytesIO que contiene el documento Word.
    """
    if not results_sequence:
        return None

    doc = Document()
    doc.add_heading("📊 Informe de Análisis Cuantitativo", 0)
    doc.add_paragraph("Este informe presenta un análisis descriptivo de las columnas numéricas y categóricas clave del archivo Excel cargado.\n")

    for item_type, title, content in results_sequence:
        if item_type == 'text':
            doc.add_paragraph(content.replace("---", "---"))
        elif item_type == 'image_bytes':
            try:
                # content es un BytesIO que contiene los bytes de la imagen
                # Resetear el puntero del BytesIO antes de usarlo
                content.seek(0) 
                doc.add_picture(content, width=Inches(6))
            except Exception as e:
                doc.add_paragraph(f"(Error al insertar imagen {title}: {e}. Asegúrate de que el archivo Word no esté abierto y tengas permisos de escritura.)")
    
    # Guardar el documento en un buffer en memoria
    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0) # Poner el puntero al principio para la lectura
    return buffer

def analyze_excel_data_streamlit(df_input):
    """
    Función principal para el análisis cuantitativo en Streamlit.
    Recibe un DataFrame de Pandas.
    Devuelve una lista de resultados (texto y figuras de matplotlib) y el BytesIO del Word.
    """
    if df_input is None or df_input.empty:
        _log_message_streamlit("No se han cargado datos válidos para el análisis.", "warning")
        return [], None, "NoData" # Retorna lista vacía, None para Word, y un estado

    _log_message_streamlit("⏳ Iniciando análisis cuantitativo...", "info")
    
    results_sequence = [] # Almacenará la secuencia de texto y objetos BytesIO de imágenes

    excel_col_names = {}
    for i, col_name_in_pandas in enumerate(df_input.columns):
        excel_letter = get_excel_column_letter(i)
        if excel_letter:
            excel_col_names[excel_letter] = col_name_in_pandas

    # Definiciones de columnas por su letra de Excel (igual que en el original)
    special_categorical_order = ['B', 'C', 'D']
    special_graph_styles = {
        'B': 'bar_vertical',
        'C': 'pie',          
        'D': 'bar_horizontal'
    }
    
    excel_numeric_cols = [
        'F', 'G', 'J', 'K', 'L', 'Q', 'T', 'V', 'W', 'X', 'AB', 'AC', 'AD', 'AE', 
        'AH', 'AI', 'AJ', 'AL', 'AM', 'AN', 'AO', 'AP', 'AQ', 'AR', 'AS', 'AT', 
        'AU', 'AV', 'AX', 'AZ', 'BA', 'BB', 'BC', 'BD', 'BE', 'BK', 'BM', 'BO', 
        'BQ', 'BV', 'BZ', 'CC'
    ]
    
    excel_phrase_analysis_cols = [
        'M', 'N', 'O', 'P', 'U', 'AY', 'BG', 'BJ', 'BL', 'BP', 'BS', 'BT', 'BX', 'BY', 'CB'
    ]
    
    excel_excluded_cols = ['Z', 'AA', 'AF', 'AG', 'BH']

    columns_to_process_final = []
    processed_pandas_cols = set() 

    def add_col_if_valid(excel_letter, graph_type, analysis_type):
        pandas_col_name = excel_col_names.get(excel_letter)
        if pandas_col_name and pandas_col_name in df_input.columns and excel_letter not in excel_excluded_cols:
            if pandas_col_name not in processed_pandas_cols:
                columns_to_process_final.append((pandas_col_name, graph_type, analysis_type))
                processed_pandas_cols.add(pandas_col_name)

    # 1. Procesar columnas especiales B, C, D en ese orden
    for excel_letter in special_categorical_order:
        add_col_if_valid(excel_letter, special_graph_styles[excel_letter], 'categorical_standard')
    
    # 2. Procesar columnas numéricas específicas
    for excel_letter in excel_numeric_cols:
        add_col_if_valid(excel_letter, 'bar_vertical', 'numeric_continuous')

    # 3. Procesar columnas de análisis de frases
    for excel_letter in excel_phrase_analysis_cols:
        add_col_if_valid(excel_letter, 'bar_horizontal', 'phrase_frequency_by_response')

    if not columns_to_process_final:
        _log_message_streamlit("El archivo Excel cargado no contiene columnas válidas para el análisis cuantitativo según las especificaciones. Asegúrate de que las columnas B, C, D, numéricas o de frases existan y no estén excluidas.", "warning")
        return [], None, "NoValidColumns"

    total_cols_to_analyze = len(columns_to_process_final)

    for idx, (pandas_col_name, graph_style, analysis_type) in enumerate(columns_to_process_final):
        col_title_display = pandas_col_name
        
        serie_raw = df_input[pandas_col_name]
        serie_cleaned = serie_raw.dropna()

        # Asegurarse de que las series numéricas sean realmente numéricas antes de operar
        if analysis_type == 'numeric_continuous':
            # Intentar convertir a numérica, coercionar errores a NaN
            serie_cleaned = pd.to_numeric(serie_cleaned, errors='coerce').dropna()
            
        if serie_cleaned.empty:
            stats_text = "(No hay datos válidos en esta columna para analizar)\n"
            analysis_explanation = "No se pudo realizar el análisis ni generar el gráfico debido a la ausencia de datos."
            results_sequence.append(('text', col_title_display, f"✨ Análisis de: {col_title_display}\n" + stats_text + "\n" + analysis_explanation + "\n" + "-" * 60 + "\n\n"))
            continue

        serie_for_plotting_original = serie_cleaned
        
        # Realizar análisis y generar texto de estadísticas
        stats_text = ""
        if analysis_type == 'phrase_frequency_by_response':
            response_counts = serie_cleaned.astype(str).value_counts() # Asegurar strings para contar
            total_responses = len(serie_cleaned)
            
            stats_text = "Frecuencia de respuestas únicas:\n"
            for response, count in response_counts.items():
                percent = (count / total_responses) * 100 if total_responses > 0 else 0
                stats_text += f"  '{response}': {count} ({percent:.2f}%)\n"
            
            analysis_explanation = generar_explicacion_analisis(col_title_display, response_counts, analysis_type, graph_style)
            serie_for_plotting = response_counts
            
        elif analysis_type == 'categorical_standard':
            value_counts = serie_cleaned.value_counts(normalize=True) * 100
            stats_text = "Frecuencia de valores:\n"
            for val, percent in value_counts.items():
                stats_text += f"  '{val}': {percent:.2f}%\n"
            analysis_explanation = generar_explicacion_analisis(col_title_display, serie_cleaned, analysis_type, graph_style)
            serie_for_plotting = serie_cleaned
        
        elif analysis_type == 'numeric_continuous':
            stats_text = (
                f"Media: {serie_cleaned.mean():.2f}, Mediana: {serie_cleaned.median():.2f}, Moda: {serie_cleaned.mode().values[0] if not serie_cleaned.mode().empty else 'N/A'}\n"
                f"Desviación estándar: {serie_cleaned.std():.2f}, Mín: {serie_cleaned.min():.2f}, Máx: {serie_cleaned.max():.2f}\n"
            )
            analysis_explanation = generar_explicacion_analisis(col_title_display, serie_cleaned, analysis_type, graph_style)
            serie_for_plotting = serie_cleaned

        full_text_block = f"✨ Análisis de: {col_title_display}\n" + stats_text + "\n" + analysis_explanation + "\n" + "-" * 60 + "\n\n"
        results_sequence.append(('text', col_title_display, full_text_block))

        # Generar gráfico y guardarlo en BytesIO
        if 'serie_for_plotting' in locals():
            fig = generar_grafico(col_title_display, serie_for_plotting, graph_style, analysis_type)
            if fig:
                # Guardar la figura en un buffer en memoria
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight", dpi=200)
                buf.seek(0) # Resetear el puntero al principio
                results_sequence.append(('image_bytes', col_title_display, buf))
            plt.close(fig) # Cerrar la figura para liberar memoria

    _log_message_streamlit("✅ Análisis cuantitativo completado.", "success")
    
    # Generar el documento Word final
    word_document_bytes = export_to_word_streamlit(results_sequence)
    
    return results_sequence, word_document_bytes, "Success"

# No se necesita el bloque if __name__ == "__main__": aquí
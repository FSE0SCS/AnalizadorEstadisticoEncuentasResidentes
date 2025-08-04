# modulo_0st.py

import pandas as pd
import numpy as np
import threading
import time
import os
import io
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import streamlit as st # Importa streamlit para las funciones de UI

# Usamos un Lock para proteger la barra de progreso de Streamlit,
# ya que no es 'thread-safe'
progress_lock = threading.Lock()

def _log_message_streamlit(message, level="info"):
    """
    Función auxiliar para mostrar mensajes en la interfaz de Streamlit.
    Adaptado de la función log_message original de Tkinter.
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
        st.write(message) # Default para cualquier otro nivel

def _process_single_excel_file(file_obj, filename, is_first_file, num_file, total_files, progress_bar_update_func, log_func):
    """
    Lógica para procesar un archivo individual de Excel.
    Se ejecuta en un hilo/proceso separado.
    
    Args:
        file_obj (UploadedFile): El objeto de archivo de Streamlit.
        filename (str): El nombre del archivo.
        is_first_file (bool): Indica si es el primer archivo a procesar.
        num_file (int): Número de archivo actual (1-based).
        total_files (int): Número total de archivos.
        progress_bar_update_func (callable): Función para actualizar la barra de progreso.
        log_func (callable): Función para registrar mensajes.
        
    Returns:
        tuple: (list, int, float) con filas procesadas, número de filas, y tiempo de procesamiento.
    """
    try:
        start_time = time.time()
        
        # Leer el archivo en un DataFrame de pandas
        df = pd.read_excel(file_obj)
        
        # Obtener los nombres de las columnas que contienen la palabra "Texto"
        text_cols = [col for col in df.columns if "Texto" in col]
        
        # Eliminar las columnas de texto, ya que la lógica de fusión no las necesita
        df_cleaned = df.drop(columns=text_cols, errors='ignore')
        
        # Eliminar filas con todos los valores NaN
        df_cleaned = df_cleaned.dropna(how='all')
        
        # Si es el primer archivo, se usa como base para las columnas
        if is_first_file:
            all_rows = df_cleaned.values.tolist()
            # Guardamos las columnas para su uso posterior en la consolidación
            columns = df_cleaned.columns.tolist()
        else:
            # Los siguientes archivos se fusionan por posición, asumiendo el mismo orden de columnas
            # Asegurarse de que el número de columnas coincide antes de la fusión
            if len(df_cleaned.columns) == len(columns):
                all_rows = df_cleaned.values.tolist()
            else:
                log_func(f"⚠️ El archivo '{filename}' tiene un número de columnas diferente. Se omitirá.", "warning")
                all_rows = []

        total_rows_processed = len(df_cleaned)
        end_time = time.time()
        time_taken = end_time - start_time
        
        # Actualizar progreso de forma segura con el lock
        with progress_lock:
            # La lógica de la barra de progreso se maneja en la función principal ahora
            pass

        return all_rows, total_rows_processed, time_taken
    
    except Exception as e:
        log_func(f"❌ Error al procesar el archivo '{filename}': {str(e)}", "error")
        return [], 0, 0.0

def process_multiple_files(uploaded_files, update_progress, log_func):
    """
    Función principal para procesar una lista de archivos subidos de manera concurrente.
    
    Args:
        uploaded_files (list): Lista de objetos UploadedFile de Streamlit.
        update_progress (callable): Callback para actualizar la barra de progreso.
        log_func (callable): Callback para mostrar mensajes.
        
    Returns:
        tuple: (DataFrame, str, str) con el DataFrame consolidado, estado y mensaje de resumen.
    """
    all_rows = []
    columns = None
    successful_files_count = 0
    total_processed_rows = 0
    total_time_taken = 0.0
    
    # Usar ThreadPoolExecutor para procesar archivos en paralelo
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        futures = []
        for i, file_obj in enumerate(uploaded_files):
            is_first_file = (i == 0)
            futures.append(
                executor.submit(
                    _process_single_excel_file, 
                    file_obj, 
                    file_obj.name, 
                    is_first_file, 
                    i + 1, 
                    len(uploaded_files), 
                    update_progress, 
                    log_func
                )
            )

        # Recopilar resultados y actualizar la barra de progreso
        for i, future in enumerate(futures):
            try:
                processed_rows, rows_count, time_taken = future.result()
                if processed_rows:
                    # El primer archivo establece las columnas
                    if i == 0:
                        df_first = pd.read_excel(uploaded_files[0])
                        text_cols_first = [col for col in df_first.columns if "Texto" in col]
                        columns = df_first.drop(columns=text_cols_first, errors='ignore').columns.tolist()
                        all_rows.extend(processed_rows)
                    else:
                        # Asegurar que el número de columnas sea el mismo
                        df_temp = pd.read_excel(uploaded_files[i])
                        text_cols_temp = [col for col in df_temp.columns if "Texto" in col]
                        if len(df_temp.drop(columns=text_cols_temp, errors='ignore').columns) == len(columns):
                            all_rows.extend(processed_rows)
                        else:
                            log_func(f"⚠️ El archivo '{uploaded_files[i].name}' tiene un número de columnas diferente. Se omitirá.", "warning")
                            continue

                    successful_files_count += 1
                    total_processed_rows += rows_count
                    total_time_taken += time_taken
                
                # Actualizar barra de progreso
                progress_percentage = int((i + 1) / len(uploaded_files) * 100)
                update_progress(progress_percentage, f"✅ Archivo {i + 1}/{len(uploaded_files)} procesado: {uploaded_files[i].name}")
                
            except Exception as e:
                log_func(f"❌ Error crítico en el resultado del hilo: {str(e)}", "error")

    if all_rows and columns:
        final_df = pd.DataFrame(all_rows, columns=columns)
        
        speed = total_processed_rows / total_time_taken if total_time_taken > 0 else 0
        
        summary_message = (
            f"🎉 PROCESAMIENTO COMPLETADO EXITOSAMENTE\n\n"
            f"• Archivos procesados: {successful_files_count}/{len(uploaded_files)}\n"
            f"• Filas totales en resultado: {len(all_rows):,}\n"
            f"• Tiempo total: {total_time_taken:.2f} segundos\n"
            f"• Velocidad promedio: {speed:.0f} filas/segundo"
        )
        
        update_progress(100, "✅ Procesamiento completado exitosamente")
        return final_df, "success", summary_message
    else:
        _log_message_streamlit("❌ No se encontraron datos válidos para procesar.", "error")
        update_progress(100, "❌ No se encontraron datos válidos.")
        return None, "error", "No se encontraron datos válidos en los archivos."

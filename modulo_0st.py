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

# No necesitamos ttkbootstrap, tkinter, etc. aquí

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

# --- FUNCIÓN CORREGIDA ---
def _process_single_excel_file(file_obj, filename, is_first_file, num_file, total_files, progress_bar_update_func, log_func):
    """
    Lógica para procesar un archivo individual de Excel.
    Se ejecuta en un hilo/proceso separado.
    
    Args:
        file_obj (UploadedFile): El objeto de archivo de Streamlit.
        filename (str): El nombre del archivo, pasado como argumento separado.
        is_first_file (bool): True si es el primer archivo.
        num_file (int): Número de archivo actual (1-based).
        total_files (int): Número total de archivos.
        progress_bar_update_func (callable): Función para actualizar la barra de progreso.
        log_func (callable): Función para registrar mensajes.
    """
    
    try:
        start_time = time.time()
        
        # Actualizar progreso (asumiendo que progress_bar_update_func es una callback)
        progress_bar_update_func(int((num_file / total_files) * 100), f"Procesando archivo {num_file} de {total_files}: {filename}")
        
        log_func(f"📂 Procesando archivo: {filename}", "info")
        
        # Leer el archivo con pandas
        df = pd.read_excel(file_obj)
        df_processed = df.copy()
        
        # Verificar si hay filas y columnas válidas
        if df_processed.empty:
            log_func(f"⚠️ El archivo '{filename}' está vacío y será omitido.", "warning")
            return [], 0, 0
            
        # Añadir columnas de metadatos
        df_processed['filename'] = filename
        df_processed['processed_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        rows_to_return = df_processed.to_dict('records')
        rows_count = len(rows_to_return)
        
        end_time = time.time()
        time_taken = end_time - start_time
        
        log_func(f"✅ Archivo '{filename}' procesado en {time_taken:.2f} segundos. {rows_count:,} filas.", "success")
        
        return rows_to_return, rows_count, time_taken
        
    except Exception as e:
        log_func(f"❌ Error al procesar el archivo '{filename}': {str(e)}", "error")
        return [], 0, 0

# --- FUNCIÓN CORREGIDA ---
def fusionar_archivos_excel_multithreaded(uploaded_files, update_progress):
    """
    Fusiona múltiples archivos Excel en un solo DataFrame de forma multihilo.
    
    Args:
        uploaded_files (list): Una lista de objetos UploadedFile de Streamlit.
        update_progress (callable): La función de la barra de progreso de Streamlit.
        
    Returns:
        tuple: (DataFrame resultante, estado, mensaje)
    """
    if not uploaded_files:
        return None, "error", "No se subieron archivos para procesar."
    
    all_rows = []
    total_processed_rows = 0
    total_time_taken = 0
    successful_files_count = 0
    
    # Creamos un Lock para proteger la barra de progreso de Streamlit
    with progress_lock:
        update_progress(0, "🚀 Iniciando procesamiento...")

    try:
        # La cantidad de hilos puede ser ajustada
        max_workers = min(os.cpu_count() or 1, len(uploaded_files))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            
            def update_progress_threadsafe(percentage, message):
                """Callback thread-safe para actualizar la barra de progreso."""
                with progress_lock:
                    update_progress(percentage, message)

            # Usamos un diccionario para mantener el orden de los archivos
            futures = {
                executor.submit(
                    _process_single_excel_file,
                    file_obj,
                    file_obj.name,  # PASAMOS EL NOMBRE COMO ARGUMENTO SEPARADO
                    i == 0, 
                    i + 1, 
                    len(uploaded_files), 
                    update_progress_threadsafe, 
                    _log_message_streamlit
                ): (file_obj, i) 
                for i, file_obj in enumerate(uploaded_files)
            }
            
            for future in futures:
                try:
                    rows, rows_count, time_taken = future.result()
                    if rows_count > 0:
                        all_rows.extend(rows)
                        total_processed_rows += rows_count
                        total_time_taken += time_taken
                        successful_files_count += 1
                except Exception as e:
                    # El error ya se registra dentro del hilo, pero lo manejamos aquí también
                    _log_message_streamlit(f"Error recuperando resultado del hilo: {str(e)}", "error")
                    
        # --- Lógica de Consolidación y Resumen ---
        if all_rows:
            # Convertir la lista de listas a DataFrame
            final_df = pd.DataFrame(all_rows)
            
            speed = total_processed_rows / total_time_taken if total_time_taken > 0 else 0
            
            summary_message = (
                f"🎉 PROCESAMIENTO COMPLETADO EXITOSAMENTE\n\n"
                f"• Archivos procesados: {successful_files_count}/{len(uploaded_files)}\n"
                f"• Filas totales en resultado: {len(all_rows):,}\n"
                f"• Tiempo total: {total_time_taken:.2f} segundos\n"
                f"• Velocidad promedio: {speed:.0f} filas/segundo"
            )
            _log_message_streamlit(summary_message, "success")
            
            update_progress(100, "✅ Procesamiento completado exitosamente")
            return final_df, "success", summary_message
        else:
            _log_message_streamlit("❌ No se encontraron datos válidos para procesar.", "error")
            update_progress(100, "❌ No se encontraron datos válidos.")
            return None, "error", "No se encontraron datos válidos en los archivos."
            
    except Exception as e:
        _log_message_streamlit(f"❌ Error crítico durante el procesamiento: {str(e)}", "error")
        update_progress(100, "❌ Error crítico.")
        return None, "error", f"Error crítico: {str(e)}"
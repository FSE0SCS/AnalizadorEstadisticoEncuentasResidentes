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

def _process_single_excel_file(file_obj, is_first_file, num_file, total_files, progress_bar_update_func, log_func):
    """
    Lógica para procesar un archivo individual de Excel.
    Se ejecuta en un hilo/proceso separado.
    """
    filename = file_obj.name # El nombre del archivo en Streamlit UploadedFile
    
    try:
        start_time = time.time()
        
        # Actualizar progreso (asumiendo que progress_bar_update_func es una callback)
        progress_value = ((num_file - 1) / total_files) * 100
        progress_bar_update_func(progress_value, f"Procesando {filename} ({num_file}/{total_files})...")
        
        log_func(f"🔍 Procesando archivo {num_file}/{total_files}: {filename}", "info")
        
        # Leer archivo Excel. io.BytesIO(file_obj.read()) es crucial para Streamlit
        df = pd.read_excel(
            io.BytesIO(file_obj.read()), # Lee el contenido binario del archivo subido
            header=None,
            usecols=range(83),  # Columnas A-CE
            dtype=str,
            na_filter=False
        )
        
        log_func(f"  📊 Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas", "info")
        
        # Ajustar columnas si es necesario
        while df.shape[1] < 83:
            df[df.shape[1]] = ''
        
        result = {
            'filename': filename,
            'valid_rows': [],
            'processed_rows_count': 0
        }
        
        # Procesar encabezados
        data_start_row_index = 1 if len(df) > 0 else 0
        if is_first_file and len(df) > 0:
            result['valid_rows'].append(df.iloc[0].values.tolist())
            log_func(f"  📋 Encabezados incluidos desde: {filename}", "info")
        
        # Procesar datos con vectorización
        if len(df) > data_start_row_index:
            rows_data = df.iloc[data_start_row_index:].values
            
            # Considera una fila válida si cualquier celda en la fila (A-CE) no está vacía o solo con espacios
            # Y también si hay datos en las columnas B-CE (índice 1 en adelante)
            # Esto intenta replicar la lógica original más fielmente
            non_empty_cells_overall = np.any((rows_data != '') & (rows_data != ' '), axis=1)
            non_empty_cells_b_to_ce = np.any((rows_data[:, 1:] != '') & (rows_data[:, 1:] != ' '), axis=1)
            
            # Una fila es válida si hay contenido en cualquier parte Y hay contenido más allá de la primera columna
            valid_mask = non_empty_cells_overall & non_empty_cells_b_to_ce
            
            valid_rows_array = rows_data[valid_mask]
            
            if len(valid_rows_array) > 0:
                result['valid_rows'].extend(valid_rows_array.tolist())
            
            result['processed_rows_count'] = len(valid_rows_array)
        
        processing_time = time.time() - start_time
        speed = result['processed_rows_count'] / processing_time if processing_time > 0 else 0
        
        log_func(
            f"  ✅ {filename}: {result['processed_rows_count']} filas válidas "
            f"en {processing_time:.2f}s ({speed:.0f} filas/s)",
            "success"
        )
        
        return result
            
    except Exception as e:
        log_func(f"  ❌ Error procesando {filename}: {str(e)}", "error")
        return {
            'filename': filename,
            'valid_rows': [],
            'processed_rows_count': 0,
            'error': str(e)
        }

def merge_excel_files_streamlit(uploaded_files):
    """
    Función principal para fusionar archivos Excel en Streamlit.
    Toma una lista de objetos UploadedFile de Streamlit.
    Devuelve un DataFrame de pandas y un mensaje de estado.
    """
    if not uploaded_files:
        return None, "warning", "No se han subido archivos Excel."

    start_total_time = time.time()
    _log_message_streamlit("🚀 Iniciando procesamiento ultra rápido...", "info")
    
    all_rows = []
    total_processed_rows = 0
    
    # Placeholder para la barra de progreso de Streamlit
    progress_text = st.empty()
    progress_bar = st.progress(0)

    def update_progress(value, text):
        progress_bar.progress(int(value))
        progress_text.text(text)
        
    def st_log_func(message, level):
        # Esta función simple solo reenvía los mensajes a la función Streamlit _log_message_streamlit
        _log_message_streamlit(message, level)
        
    try:
        # Preparamos los argumentos para el procesamiento paralelo
        processing_args = []
        for i, file_obj in enumerate(uploaded_files):
            is_first_file = (i == 0)
            processing_args.append((file_obj, is_first_file, i + 1, len(uploaded_files), update_progress, st_log_func))
        
        results = []
        if len(uploaded_files) == 1:
            _log_message_streamlit("📊 Procesando archivo único...", "info")
            results.append(_process_single_excel_file(processing_args[0][0], processing_args[0][1], processing_args[0][2], processing_args[0][3], processing_args[0][4], processing_args[0][5]))
        else:
            _log_message_streamlit("📊 Procesando primer archivo (encabezados)...", "info")
            # Procesar el primer archivo secuencialmente para asegurar la obtención de encabezados
            first_file_result = _process_single_excel_file(processing_args[0][0], processing_args[0][1], processing_args[0][2], processing_args[0][3], processing_args[0][4], processing_args[0][5])
            results.append(first_file_result)

            _log_message_streamlit("⚡ Procesando archivos restantes en paralelo...", "info")
            # Los archivos restantes en paralelo
            # Usar max_workers para evitar sobrecargar, 4 es un buen valor por defecto
            with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
                # Cuidado: no pasar objetos de archivo directamente si se van a leer múltiples veces en hilos.
                # Aquí, la lectura `.read()` ya se hace dentro de `_process_single_excel_file`,
                # así que cada hilo obtendrá su propia copia de los bytes del archivo.
                # Para evitar re-leer bytes en cada re-ejecución, podrías pre-leer los bytes
                # y pasarlos como io.BytesIO a cada llamada.
                
                # Para un solo paso, la lectura directa de file_obj.read() dentro del worker es aceptable.
                # Si esto fuera un problema de rendimiento, optimizaríamos pre-leyendo.
                
                # Mapear los argumentos a la función
                parallel_results = list(executor.map(
                    lambda p_args: _process_single_excel_file(*p_args),
                    processing_args[1:]
                ))
                results.extend(parallel_results)
        
        # Consolidar resultados
        successful_files_count = 0
        for result in results:
            if result and 'error' not in result:
                all_rows.extend(result['valid_rows'])
                total_processed_rows += result['processed_rows_count']
                successful_files_count += 1
            elif result and 'error' in result:
                _log_message_streamlit(f"❌ Error en {result['filename']}: {result['error']}", "error")
        
        total_time_taken = time.time() - start_total_time
        
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
        return None, "error", f"Error crítico durante el procesamiento:\n{str(e)}"

# No se necesita el bloque if __name__ == "__main__": aquí, ya que será importado por app.py
# prompt_creator_app_st.py

import sqlite3
import streamlit as st
import os

# Nombre de la base de datos
DB_NAME = "prompt_data.db"

# --- Funciones de Conexión y Consulta a la Base de Datos ---

@st.cache_resource
def get_db_connection():
    """
    Establece y devuelve una conexión a la base de datos SQLite.
    Usa st.cache_resource para asegurar que la conexión se mantenga.
    """
    try:
        # Asegúrate de que la DB esté en el mismo directorio o especifica la ruta completa
        conn = sqlite3.connect(DB_NAME)
        conn.row_factory = sqlite3.Row # Para acceder a las columnas por nombre
        st.success(f"Conexión a la base de datos '{DB_NAME}' establecida.")
        return conn
    except sqlite3.Error as e:
        st.error(f"Error al conectar a la base de datos: {e}")
        st.warning(f"Asegúrate de que el archivo '{DB_NAME}' esté en el mismo directorio que la aplicación Streamlit.")
        return None

def get_all_verbs():
    """Obtiene todos los verbos disponibles de la base de datos."""
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT verbo FROM Verbos ORDER BY verbo")
            verbs = [row['verbo'] for row in cursor.fetchall()]
            return ["Selecciona un verbo"] + verbs # Añadir opción predeterminada
        except sqlite3.Error as e:
            st.error(f"Error al obtener verbos: {e}")
    return ["Error al cargar verbos"]

def get_role_categories():
    """Obtiene todas las categorías de rol disponibles."""
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT categoria_rol FROM Roles ORDER BY categoria_rol")
            categories = [row['categoria_rol'] for row in cursor.fetchall()]
            return ["Selecciona una categoría"] + categories # Añadir opción predeterminada
        except sqlite3.Error as e:
            st.error(f"Error al obtener categorías de rol: {e}")
    return ["Error al cargar categorías"]

def get_roles_by_category(category):
    """Obtiene roles basados en una categoría seleccionada."""
    conn = get_db_connection()
    if conn and category and category != "Selecciona una categoría":
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT rol FROM Roles WHERE categoria_rol = ? ORDER BY rol", (category,))
            roles = [row['rol'] for row in cursor.fetchall()]
            return ["Selecciona un rol"] + roles # Añadir opción predeterminada
        except sqlite3.Error as e:
            st.error(f"Error al obtener roles por categoría: {e}")
    return ["Selecciona un rol"]

# --- Lógica de Generación de Prompt ---

def generate_ai_prompt(selected_verb, selected_role, additional_context):
    """
    Genera el prompt de IA combinado.
    """
    prompt_parts = []
    if selected_verb and selected_verb != "Selecciona un verbo":
        prompt_parts.append(selected_verb)
    
    if selected_role and selected_role != "Selecciona un rol":
        prompt_parts.append(f"como {selected_role}")
    
    if additional_context:
        prompt_parts.append(f"y {additional_context}")
    
    if prompt_parts:
        final_prompt = " ".join(prompt_parts).strip() + "."
        return final_prompt
    return "Tu prompt aparecerá aquí."

# --- Nota para la implementación en app.py ---
# Las funciones `get_all_verbs`, `get_role_categories`, `get_roles_by_category`
# y `generate_ai_prompt` serán llamadas directamente desde `app.py` para construir
# la interfaz de usuario y la lógica de interacción del creador de prompts.
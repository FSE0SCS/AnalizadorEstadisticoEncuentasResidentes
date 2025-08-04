# prompt_creator_app_st.py

import sqlite3
import streamlit as st
import os

# Nombre de la base de datos
DB_NAME = "prompt_data.db"

# --- NUEVA FUNCIÓN PARA CREAR TABLAS Y DATOS INICIALES ---
def _create_tables_and_populate_data(conn):
    """
    Crea las tablas 'Verbos' y 'Roles' si no existen, y las puebla
    con datos de ejemplo si están vacías.
    """
    try:
        cursor = conn.cursor()

        # Crear la tabla 'Verbos'
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Verbos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                verbo TEXT NOT NULL UNIQUE
            )
        """)
        
        # Insertar verbos de ejemplo si la tabla está vacía
        cursor.execute("SELECT COUNT(*) FROM Verbos")
        if cursor.fetchone()[0] == 0:
            verbos_ejemplo = [
                ('Analizar',), ('Resumir',), ('Generar',), ('Escribir',),
                ('Explicar',), ('Traducir',), ('Clasificar',), ('Revisar',)
            ]
            cursor.executemany("INSERT INTO Verbos (verbo) VALUES (?)", verbos_ejemplo)

        # Crear la tabla 'Roles'
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Roles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                categoria_rol TEXT NOT NULL,
                rol TEXT NOT NULL,
                descripcion TEXT
            )
        """)
        
        # Insertar roles de ejemplo si la tabla está vacía
        cursor.execute("SELECT COUNT(*) FROM Roles")
        if cursor.fetchone()[0] == 0:
            roles_ejemplo = [
                ('Profesional', 'Experto en Marketing Digital', 'Un experto en marketing digital con conocimientos en SEO, SEM y redes sociales.'),
                ('Profesional', 'Científico de Datos', 'Un científico de datos especializado en análisis estadístico y machine learning.'),
                ('Creativo', 'Escritor de Novelas', 'Un autor creativo con la capacidad de desarrollar personajes y tramas complejas.'),
                ('Creativo', 'Guionista de Cine', 'Un guionista con experiencia en la estructura narrativa de películas y series de televisión.'),
                ('Académico', 'Profesor Universitario', 'Un profesor con conocimiento profundo de su materia y habilidad para explicar conceptos complejos.'),
                ('Académico', 'Historiador', 'Un historiador con la capacidad de analizar eventos pasados y sus implicaciones.'),
            ]
            cursor.executemany("INSERT INTO Roles (categoria_rol, rol, descripcion) VALUES (?, ?, ?)", roles_ejemplo)

        conn.commit()
        st.info("Tablas de la base de datos verificadas y pobladas con datos de ejemplo.")

    except sqlite3.Error as e:
        st.error(f"Error al verificar/crear tablas: {e}")


# --- Funciones de Conexión y Consulta a la Base de Datos ---

@st.cache_resource
def get_db_connection():
    """
    Establece y devuelve una conexión a la base de datos SQLite.
    Usa st.cache_resource para asegurar que la conexión se mantenga.
    """
    try:
        conn = sqlite3.connect(DB_NAME)
        conn.row_factory = sqlite3.Row # Para acceder a las columnas por nombre
        _create_tables_and_populate_data(conn) # Llama a la nueva función aquí
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
    return ["Selecciona un verbo"]

def get_role_categories():
    """Obtiene todas las categorías de rol disponibles."""
    conn = get_db_connection()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT categoria_rol FROM Roles ORDER BY categoria_rol")
            categories = [row['categoria_rol'] for row in cursor.fetchall()]
            return ["Selecciona una categoría"] + categories
        except sqlite3.Error as e:
            st.error(f"Error al obtener categorías de rol: {e}")
    return ["Selecciona una categoría"]

def get_roles_by_category(category):
    """Obtiene los roles para una categoría específica."""
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
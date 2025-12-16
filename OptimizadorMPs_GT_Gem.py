# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 22:15:30 2025

@author: Usuario
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO

# ==============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ==============================================================================
st.set_page_config(page_title="MicroPile Opt V3", layout="wide", page_icon="🏗️")

# Estilos CSS para intentar imitar el look limpio de la versión React
st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; }
    h1 { color: #1e3a8a; }
    h2 { color: #1e40af; font-size: 1.5rem; }
    h3 { color: #374151; font-size: 1.2rem; }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; }
    .stButton>button:hover { border-color: #2563eb; color: #2563eb; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; }
</style>
""", unsafe_allow_html=True)

# --- CONSTANTES GLOBALES ---
DIAMETROS_COM = {100: 1.00, 115: 0.95, 130: 0.93, 150: 0.90, 200: 0.85}
LISTA_D = sorted(list(DIAMETROS_COM.keys()))
COSTO_PERF_BASE = 100
FACTOR_CO2_CEMENTO = 0.90
FACTOR_CO2_PERF = 15.0
FACTOR_CO2_ACERO = 1.85
DENSIDAD_ACERO = 7850.0
DENSIDAD_CEMENTO = 3150.0
FY_ACERO_KPA = 500000.0

# ==============================================================================
# 1. SISTEMA DE LOGIN (Session State)
# ==============================================================================
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
    st.session_state['user_info'] = {}

# Inicializar capas por defecto si no existen
if 'layers' not in st.session_state:
    st.session_state['layers'] = [
        {"name": "Relleno / Arcilla Blanda", "thickness": 3.0, "qs": 40.0, "f_exp": 1.1, "color": "#dbeafe"},
        {"name": "Arcilla Firme / Limo", "thickness": 5.0, "qs": 80.0, "f_exp": 1.2, "color": "#fef3c7"},
        {"name": "Estrato Resistente", "thickness": 10.0, "qs": 150.0, "f_exp": 1.3, "color": "#fee2e2"}
    ]

# Inicializar resultados globales para compartir entre pestañas
if 'global_results' not in st.session_state:
    st.session_state['global_results'] = None

def login_screen():
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("<div style='text-align: center; padding: 40px; background-color: #f8fafc; border-radius: 15px; border: 1px solid #e2e8f0;'>", unsafe_allow_html=True)
        st.title("🔒 Acceso de Ingeniería")
        st.markdown("### Sistema de Optimización de Micropilotes")
        
        with st.form("login_form"):
            nombre = st.text_input("Nombre Completo")
            email = st.text_input("Correo Electrónico")
            empresa = st.text_input("Empresa")
            cargo = st.selectbox("Cargo", ["Ingeniero Geotecnista", "Ingeniero Estructural", "Constructor/Residente"])
            acepto = st.checkbox("Acepto los términos de uso técnico y registro.")
            
            submitted = st.form_submit_button("🚀 INGRESAR AL SISTEMA")
            
            if submitted:
                if nombre and email and empresa and acepto:
                    st.session_state['logged_in'] = True
                    st.session_state['user_info'] = {'nombre': nombre, 'email': email, 'empresa': empresa}
                    st.rerun()
                else:
                    st.error("Por favor complete todos los campos y acepte los términos.")
        st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# 2. FUNCIONES DE CÁLCULO Y GRÁFICOS
# ==============================================================================

def draw_stratigraphy_profile(layers, water_table=None):
    """Dibuja el perfil estratigráfico usando Matplotlib"""
    fig, ax = plt.subplots(figsize=(4, 8))
    
    current_depth = 0
    total_depth = sum(l['thickness'] for l in layers)
    max_depth = max(total_depth, 15) * 1.1
    
    for layer in layers:
        # Dibujar rectángulo
        rect = patches.Rectangle((0, current_depth), 10, layer['thickness'], 
                               linewidth=1, edgecolor='white', facecolor=layer['color'])
        ax.add_patch(rect)
        
        # Texto centrado
        mid_y = current_depth + layer['thickness']/2
        ax.text(5, mid_y, f"{layer['name']}\nQs={int(layer['qs'])} kPa\nF.Exp={layer['f_exp']}", 
                ha='center', va='center', fontsize=8, color='#334155', fontweight='bold')
        
        # Línea de cota
        current_depth += layer['thickness']
        ax.axhline(y=current_depth, color='gray', linestyle=':', linewidth=0.5)
        ax.text(10.2, current_depth, f"{current_depth:.1f}m", va='center', fontsize=8)

    # Nivel Freático
    if water_table is not None and water_table > 0:
        ax.axhline(y=water_table, color='blue', linestyle='--', linewidth=2)
        ax.text(10.2, water_table, "N.F.", color='blue', fontsize=9, fontweight='bold', va='bottom')
        # Triangulito
        triangle = patches.Polygon([[8, water_table], [9, water_table], [8.5, water_table+0.5]], closed=True, color='blue')
        ax.add_patch(triangle)

    ax.set_ylim(max_depth, 0) # Invertir eje Y
    ax.set_xlim(0, 10)
    ax.set_xticks([])
    ax.set_ylabel("Profundidad (m)")
    ax.set_title("Perfil Estratigráfico")
    
    return fig

def draw_spt_qs_graphs(data, k_factor):
    """Dibuja N-SPT y Qs lado a lado"""
    z = [d['z'] for d in data]
    n = [d['n'] for d in data]
    qs = [min(d['n'] * k_factor, 250) for d in data] # Calculo al vuelo
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 8), sharey=True)
    
    # N-SPT
    ax1.plot(n, z, 'o-', color='#2563eb', linewidth=2)
    ax1.set_ylim(max(z)+2, 0)
    ax1.set_xlabel("N-SPT (Golpes)")
    ax1.set_ylabel("Profundidad (m)")
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_title("N-SPT")
    
    # Qs
    ax2.plot(qs, z, 's-', color='#dc2626', linewidth=2)
    ax2.set_xlabel("Adherencia Qs (kPa)")
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_title(f"Qs Est. (K={k_factor})")
    
    return fig

def solve_optimization(load_ton, fs_req, wc_ratio, min_micros, max_micros, min_d, max_d, layers):
    carga_req_kn = load_ton * 9.81
    solutions = []
    
    # Filtrar diámetros
    diametros_validos = [d for d in LISTA_D if min_d <= d <= max_d]
    
    for D_mm in diametros_validos:
        D_m = D_mm / 1000.0
        eficiencia = DIAMETROS_COM.get(D_mm, 0.85)
        
        for N in range(min_micros, max_micros + 1):
            q_act_pilote = carga_req_kn / N
            q_req_geo = q_act_pilote * fs_req
            
            # Buscar L óptima
            for L in np.arange(5.0, 40.5, 0.5):
                q_ult = 0
                vol_exp_one = 0
                area_perf = np.pi * (D_m/2)**2
                acc_depth = 0
                
                # Integración por capas
                for layer in layers:
                    layer_top = acc_depth
                    layer_bottom = acc_depth + layer['thickness']
                    acc_depth += layer['thickness']
                    
                    start = max(0, layer_top)
                    end = min(L, layer_bottom)
                    seg_len = max(0, end - start)
                    
                    if seg_len > 0:
                        # ECUACIÓN ACTUALIZADA: Di = D * F_exp (Lineal)
                        d_eff = D_m * layer['f_exp']
                        area_lat = np.pi * d_eff * seg_len
                        q_ult += area_lat * layer['qs']
                        
                        vol_exp_one += area_perf * seg_len * layer['f_exp']
                
                # Extensión último estrato si L > prof. total
                if L > acc_depth:
                    extra_len = L - acc_depth
                    last_layer = layers[-1]
                    d_eff = D_m * last_layer['f_exp']
                    q_ult += (np.pi * d_eff * extra_len) * last_layer['qs']
                    vol_exp_one += area_perf * extra_len * last_layer['f_exp']
                
                if q_ult >= q_req_geo:
                    # Encontrado solución
                    vol_exp_total = vol_exp_one * N
                    costo_idx = (L * N * COSTO_PERF_BASE) / eficiencia
                    
                    # CO2
                    area_acero = q_act_pilote / FY_ACERO_KPA
                    vol_acero = area_acero * L * N
                    peso_acero = vol_acero * DENSIDAD_ACERO
                    
                    vol_grout_neto = max(0, vol_exp_total - vol_acero)
                    peso_cemento = vol_grout_neto * (1000 / (wc_ratio + 1/3.15))
                    
                    metros_perf = L * N
                    co2 = (peso_acero * FACTOR_CO2_ACERO + peso_cemento * FACTOR_CO2_CEMENTO + metros_perf * FACTOR_CO2_PERF) / 1000
                    
                    solutions.append({
                        "D_mm": D_mm,
                        "N": N,
                        "L": float(L),
                        "Perf_Total": float(L*N),
                        "FS": q_ult / q_act_pilote,
                        "Q_adm": q_ult / fs_req / 9.81, # Ton
                        "Q_act": q_act_pilote / 9.81,   # Ton
                        "Vol_Grout": vol_exp_total,
                        "CO2": co2,
                        "Costo_Idx": costo_idx
                    })
                    break # Salir del loop de L, pasar al siguiente N
                    
    return pd.DataFrame(solutions).sort_values("Costo_Idx")

# ==============================================================================
# 3. INTERFAZ PRINCIPAL
# ==============================================================================

def main_app():
    # Sidebar Global
    with st.sidebar:
        st.info(f"👤 **{st.session_state['user_info']['nombre']}**\n\n{st.session_state['user_info']['cargo']}")
        if st.button("Cerrar Sesión"):
            st.session_state['logged_in'] = False
            st.rerun()
    
    # Header
    st.title("🏗️ Optimizador de Micropilotes")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["1. Info Geotécnica", "2. Diseño", "3. Dados / Cabezales"])
    
    # --- TAB 1: GEOTECNIA ---
    with tab1:
        col_input, col_viz = st.columns([1, 2])
        
        with col_input:
            st.subheader("Datos de Campo (SPT)")
            
            # Editor de Datos SPT
            if 'spt_data' not in st.session_state:
                st.session_state['spt_data'] = pd.DataFrame([
                    {"z": 1.5, "n": 4}, {"z": 3.0, "n": 7}, {"z": 4.5, "n": 12},
                    {"z": 6.0, "n": 15}, {"z": 7.5, "n": 22}, {"z": 9.0, "n": 28},
                    {"z": 10.5, "n": 35}, {"z": 12.0, "n": 42}, {"z": 15.0, "n": 50}
                ])
            
            edited_spt = st.data_editor(st.session_state['spt_data'], num_rows="dynamic", use_container_width=True)
            st.session_state['spt_data'] = edited_spt
            
            k_factor = st.slider("Factor de Correlación (K)", 1.0, 10.0, 3.5, 0.5)
            st.caption("Qs ≈ K · N (Limitado a 250 kPa)")
            
            nf_depth = st.number_input("Nivel Freático (m)", 0.0, 50.0, 2.0, 0.5)
            
            st.divider()
            st.subheader("Definición de Estratos")
            
            # Editor de Capas (Usando Dataframe para simular la interactividad)
            layers_df = pd.DataFrame(st.session_state['layers'])
            edited_layers = st.data_editor(layers_df, num_rows="dynamic", use_container_width=True,
                                           column_config={
                                               "color": st.column_config.ColorColumn("Color"),
                                               "name": "Nombre",
                                               "thickness": st.column_config.NumberColumn("Espesor (m)", min_value=0.1),
                                               "qs": st.column_config.NumberColumn("Qs (kPa)", min_value=0),
                                               "f_exp": st.column_config.NumberColumn("F. Exp", min_value=1.0, max_value=3.0, step=0.1)
                                           })
            # Actualizar estado de capas
            st.session_state['layers'] = edited_layers.to_dict('records')

        with col_viz:
            st.subheader("Modelo Geotécnico Integrado")
            
            # Generar gráfico combinado
            c_prof, c_graphs = st.columns([1, 2])
            
            with c_prof:
                fig_strat = draw_stratigraphy_profile(st.session_state['layers'], nf_depth)
                st.pyplot(fig_strat)
            
            with c_graphs:
                fig_curves = draw_spt_qs_graphs(st.session_state['spt_data'].to_dict('records'), k_factor)
                st.pyplot(fig_curves)
            
            with st.expander("Ver Tablas de Referencia (FHWA / Bustamante)"):
                st.markdown("""
                **FHWA (NHI-05-039)**: [Enlace al Manual](https://rosap.ntl.bts.gov/view/dot/50231)
                
                | Tipo de Suelo (Tipo A) | Qs Típico (kPa) |
                | :--- | :--- |
                | Arcilla Blanda / Limo | 20 - 60 |
                | Arcilla Media | 40 - 90 |
                | Arena Media/Densa | 100 - 250 |
                | Roca / Grava | 200 - 500+ |
                """)

    # --- TAB 2: DISEÑO ---
    with tab2:
        # Header Ecuación
        st.markdown(r"""
        <div style="background-color:white; padding:10px; border-bottom:1px solid #ddd; margin-bottom:20px; text-align:center;">
            <strong>Ecuación de Diseño:</strong> $Q_{ult} = \pi \cdot \sum ( D_{nom} \cdot f_{exp,i} \cdot L_i \cdot q_{s,i} )$ &nbsp;&nbsp;|&nbsp;&nbsp; $FS = Q_{ult} / Q_{act} \ge FS_{req}$
        </div>
        """, unsafe_allow_html=True)
        
        col_params, col_res = st.columns([1, 3])
        
        with col_params:
            st.subheader("Configuración")
            load_ton = st.number_input("Carga Cabezal (Ton)", value=120.0)
            fs_req = st.number_input("FS Requerido", value=2.0, step=0.1)
            wc_ratio = st.number_input("Rel. A/C Grout", value=0.50, step=0.05)
            
            c_d1, c_d2 = st.columns(2)
            min_d = c_d1.selectbox("Min Ø (mm)", LISTA_D, index=0)
            max_d = c_d2.selectbox("Max Ø (mm)", LISTA_D, index=len(LISTA_D)-1)
            
            c_n1, c_n2 = st.columns(2)
            min_n = c_n1.number_input("Min Cant.", 1, 20, 1)
            max_n = c_n2.number_input("Max Cant.", 1, 20, 10)
            
            calc_btn = st.button("🚀 CALCULAR DISEÑO", type="primary")
        
        with col_res:
            if calc_btn:
                with st.spinner("Optimizando miles de combinaciones..."):
                    df_res = solve_optimization(load_ton, fs_req, wc_ratio, min_n, max_n, min_d, max_d, st.session_state['layers'])
                    
                    if df_res.empty:
                        st.error("No se encontraron soluciones viables dentro de los rangos establecidos.")
                    else:
                        st.session_state['global_results'] = df_res # Guardar para Tab 3
                        
                        # KPI Cards
                        best = df_res.iloc[0]
                        k1, k2, k3, k4 = st.columns(4)
                        k1.metric("Mejor Config", f"{int(best['N'])} x Ø{int(best['D_mm'])}")
                        k2.metric("Longitud", f"{best['L']} m", f"Total: {best['Perf_Total']:.0f}m")
                        k3.metric("Grout", f"{best['Vol_Grout']:.1f} m³")
                        k4.metric("Huella CO2", f"{best['CO2']:.1f} Ton")
                        
                        # Tabla Interactiva
                        st.subheader("Top Alternativas")
                        
                        # Formato para visualización
                        display_cols = ["D_mm", "N", "L", "Perf_Total", "FS", "Q_adm", "Q_act", "Vol_Grout", "CO2"]
                        
                        # Selección múltiple
                        df_display = df_res[display_cols].head(15).copy()
                        df_display["Seleccionar"] = False # Checkbox column simulation
                        
                        edited_df = st.data_editor(
                            df_display,
                            column_config={
                                "Seleccionar": st.column_config.CheckboxColumn("Ver", help="Seleccione para ver en Dados"),
                                "CO2": st.column_config.ProgressColumn("Huella CO2", format="%.1f T", min_value=0, max_value=max(df_display["CO2"])),
                                "Perf_Total": st.column_config.NumberColumn("Perf. Total", format="%.0f m"),
                            },
                            hide_index=True,
                            use_container_width=True
                        )
                        
                        # Guardar selección para Tab 3
                        selected_indices = edited_df[edited_df["Seleccionar"]].index.tolist()
                        if not selected_indices:
                            selected_indices = [0, 1, 2] if len(df_res) >=3 else list(range(len(df_res)))
                        
                        st.session_state['selected_indices'] = selected_indices
                        
                        # Botón Descarga CSV (Formato Excel ES)
                        csv = df_res.to_csv(sep=';', decimal=',', index=False).encode('utf-8-sig')
                        st.download_button("📥 Descargar CSV Completo", data=csv, file_name="optimizacion_micropilotes.csv", mime="text/csv")

    # --- TAB 3: DADOS ---
    with tab3:
        if st.session_state['global_results'] is None:
            st.info("Primero ejecute el cálculo en la pestaña 'Diseño'.")
        else:
            df = st.session_state['global_results']
            indices = st.session_state.get('selected_indices', [0, 1, 2])
            
            st.subheader(f"Esquema de Dados ({len(indices)} seleccionados)")
            
            cols_ui = st.columns(3)
            for i, idx in enumerate(indices):
                if idx in df.index:
                    row = df.loc[idx]
                    
                    # Cálculos geométricos simples para dado
                    N = int(row['N'])
                    D = row['D_mm']/1000
                    S = max(0.75, 3*D)
                    Borde = max(0.30, 1.5*D)
                    
                    # Matplotlib visualización simple del dado
                    fig, ax = plt.subplots(figsize=(3, 3))
                    ax.set_aspect('equal')
                    
                    # Lógica de grid simple
                    grid_cols = int(np.ceil(np.sqrt(N)))
                    grid_rows = int(np.ceil(N/grid_cols))
                    
                    W = (grid_cols-1)*S + 2*Borde
                    L = (grid_rows-1)*S + 2*Borde
                    
                    # Dibujar Dado
                    rect = patches.Rectangle((0,0), W, L, facecolor='#e2e8f0', edgecolor='black')
                    ax.add_patch(rect)
                    
                    # Dibujar Pilotes
                    for p_i in range(N):
                        r = p_i // grid_cols
                        c = p_i % grid_cols
                        cx = Borde + c*S
                        cy = Borde + r*S
                        circ = patches.Circle((cx, cy), D/2, facecolor='#1e293b')
                        ax.add_patch(circ)
                    
                    ax.set_xlim(-0.5, W+0.5)
                    ax.set_ylim(-0.5, L+0.5)
                    ax.axis('off')
                    ax.set_title(f"{N} x Ø{int(row['D_mm'])}mm", fontsize=10)
                    
                    # Mostrar en columna
                    with cols_ui[i % 3]:
                        st.pyplot(fig)
                        st.caption(f"**Vol:** {(W*L*0.6):.1f}m³ | **Dim:** {W:.1f}x{L:.1f}x0.6m")

# MAIN ENTRY POINT
if __name__ == "__main__":
    if not st.session_state['logged_in']:
        login_screen()
    else:
        main_app()
```

### ¿Cómo ejecutar esta versión en Python?

1.  **Instala las dependencias:**
    Guarda el siguiente contenido en un archivo llamado `requirements.txt`:
    ```txt
    streamlit
    pandas
    numpy
    matplotlib
    ```
    Luego ejecuta en tu terminal:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Guarda el código:**
    Copia el bloque de código de arriba y guárdalo como `app.py`.

3.  **Ejecuta:**
    En tu terminal (dentro de la carpeta donde guardaste el archivo):
    ```bash
    streamlit run app.py
import streamlit as st
import math
import json
from io import StringIO
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import re

# =========================
# Configuración única
# =========================
st.set_page_config(
    page_title="MVP Micropilotes | Diseño & SPT",
    layout="wide",
    page_icon="🏗️"
)

# =========================
# Constantes (MVP)
# =========================
DIAMETROS_COM = {0.100: 1.00, 0.115: 0.95, 0.150: 0.90, 0.200: 0.85}
LISTA_D = sorted(DIAMETROS_COM.keys())
MIN_MICROS, MAX_MICROS = 1, 15
RANGO_L = range(5, 60)
COSTO_PERF_BASE = 100

FACTOR_CO2_CEMENTO = 0.90
FACTOR_CO2_PERF = 15.0
FACTOR_CO2_ACERO = 1.85
DENSIDAD_ACERO = 7850.0
DENSIDAD_CEMENTO = 3150.0
FY_ACERO_KPA = 500000.0  # 500 MPa

COLORES_ESTRATOS = ["#D7BDE2", "#A9CCE3", "#A3E4D7", "#F9E79F", "#F5B7B1",
                    "#D2B4DE", "#AED6F1", "#A2D9CE", "#F7DC6F", "#F1948A"]

DATA_DIR = Path("data"); DATA_DIR.mkdir(exist_ok=True)
RUNS_DIR = Path("runs"); RUNS_DIR.mkdir(exist_ok=True)
LEADS_FILE = DATA_DIR / "leads.csv"

# =========================
# Estado de sesión
# =========================
if "usuario_registrado" not in st.session_state:
    st.session_state.usuario_registrado = False
if "datos_usuario" not in st.session_state:
    st.session_state.datos_usuario = {}

# =========================
# Utilidades
# =========================
def is_valid_email(email: str) -> bool:
    pat = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
    return bool(re.match(pat, email.strip()))

def save_lead(datos: dict) -> None:
    df_new = pd.DataFrame([datos])
    if LEADS_FILE.exists():
        df_old = pd.read_csv(LEADS_FILE)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_csv(LEADS_FILE, index=False)

def save_run(payload: dict) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fp = RUNS_DIR / f"run_{ts}.json"
    fp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return fp

# =========================
# Cálculos (núcleo MVP)
# =========================
def calc_capacidad_individual(D, L_total, estratos):
    Q_ult = 0.0
    z_actual = 0.0
    for e in estratos:
        if z_actual >= L_total:
            break
        z_fondo = min(e["z_fin"], L_total)
        dz = z_fondo - z_actual
        if dz > 0:
            area_lateral = math.pi * D * dz
            Q_ult += area_lateral * e["qs"]  # (m2 * kPa) -> kN aprox (por consistencia simplificada)
        z_actual = z_fondo
    return Q_ult

def calc_volumenes_grout(D, L_total, estratos):
    vol_teo = 0.0
    vol_exp = 0.0
    z_actual = 0.0
    area_perf = math.pi * (D/2)**2
    for e in estratos:
        if z_actual >= L_total:
            break
        z_fondo = min(e["z_fin"], L_total)
        dz = z_fondo - z_actual
        if dz > 0:
            v_teo = area_perf * dz
            v_exp = v_teo * e["f_exp"]
            vol_teo += v_teo
            vol_exp += v_exp
        z_actual = z_fondo
    return vol_teo, vol_exp

def peso_cemento_por_m3(wc):
    # kg cemento por m3 (modelo simple coherente con tu planteamiento)
    return 1.0 / (wc/1000.0 + 1.0/DENSIDAD_CEMENTO)

def calc_co2(L, N, vol_grout_exp_total, Q_act_por_pilote_kN, wc):
    # Área acero requerida ~ Carga/Fy
    area_acero_m2 = Q_act_por_pilote_kN / FY_ACERO_KPA
    vol_acero = area_acero_m2 * L * N
    peso_acero = vol_acero * DENSIDAD_ACERO

    vol_grout_neto = max(0.0, vol_grout_exp_total - vol_acero)
    peso_cemento = vol_grout_neto * peso_cemento_por_m3(wc)

    metros_perf = L * N
    co2_kg = (peso_acero * FACTOR_CO2_ACERO
              + peso_cemento * FACTOR_CO2_CEMENTO
              + metros_perf * FACTOR_CO2_PERF)
    return co2_kg / 1000.0, area_acero_m2 * 10000.0  # ton CO2, cm2 acero

def perfil_transferencia(D, L, estratos):
    z_points = [0.0]
    q_points = [0.0]
    z_actual = 0.0
    q_acum = 0.0
    for e in estratos:
        if z_actual >= L:
            break
        z_fondo = min(e["z_fin"], L)
        dz = z_fondo - z_actual
        if dz > 0:
            q_tramo = (math.pi * D * dz) * e["qs"]
            q_acum += q_tramo
            z_points.append(z_fondo)
            q_points.append(q_acum)
        z_actual = z_fondo
    return z_points, q_points

def optimizar_diseno(estratos, carga_ton, fs_req, wc):
    carga_kN = carga_ton * 9.81
    resultados = []
    for D in LISTA_D:
        eficiencia = DIAMETROS_COM[D]
        for N in range(MIN_MICROS, MAX_MICROS + 1):
            Q_act_pilote = carga_kN / N
            Q_req_geo = Q_act_pilote * fs_req
            for L in RANGO_L:
                Q_ult = calc_capacidad_individual(D, L, estratos)
                if Q_ult >= Q_req_geo:
                    fs_calc = Q_ult / Q_act_pilote
                    v_teo_m, v_exp_m = calc_volumenes_grout(D, L, estratos)
                    v_exp_tot = v_exp_m * N

                    costo_idx = (L * N * COSTO_PERF_BASE) / eficiencia
                    co2_ton, acero_cm2 = calc_co2(L, N, v_exp_tot, Q_act_pilote, wc)

                    resultados.append({
                        "D_val": D,
                        "D_mm": int(D * 1000),
                        "N": N,
                        "L_m": L,
                        "Perf_m": L * N,
                        "FS": fs_calc,
                        "Grout_m3": v_exp_tot,
                        "Costo_Idx": costo_idx,
                        "CO2_ton": co2_ton,
                        "Qact_T": Q_act_pilote / 9.81,
                        "Qadm_T": (Q_ult / fs_req) / 9.81,
                        "Acero_req_cm2": acero_cm2
                    })
                    break
    if not resultados:
        return pd.DataFrame()
    df = pd.DataFrame(resultados).sort_values("Costo_Idx", ascending=True).reset_index(drop=True)
    return df

# =========================
# UI: Registro (Lead gate)
# =========================
def mostrar_registro():
    st.title("🏗️ MVP | Micropilotes")
    st.markdown("Acceso mediante registro (captura de leads).")

    with st.form("registro"):
        c1, c2 = st.columns(2)
        nombre = c1.text_input("Nombre completo")
        empresa = c2.text_input("Empresa / Universidad")
        email = st.text_input("Correo corporativo")
        cargo = st.selectbox(
            "Cargo",
            ["Ingeniero Geotecnista", "Ingeniero Estructural", "Constructor/Residente", "Estudiante", "Otro"]
        )
        acepto = st.checkbox("Acepto recibir información técnica relacionada.")
        submit = st.form_submit_button("Ingresar")

    if submit:
        if not (nombre and empresa and email):
            st.error("Complete nombre, empresa y correo.")
            return
        if not is_valid_email(email):
            st.error("Correo no válido. Verifique el formato.")
            return
        if not acepto:
            st.error("Para continuar debe aceptar el consentimiento.")
            return

        lead = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "nombre": nombre.strip(),
            "empresa": empresa.strip(),
            "email": email.strip().lower(),
            "cargo": cargo
        }
        save_lead(lead)

        st.session_state.usuario_registrado = True
        st.session_state.datos_usuario = lead
        st.success("Acceso concedido.")
        st.rerun()

# =========================
# UI: Aplicación principal
# =========================
def app_principal():
    with st.sidebar:
        st.success(f"Sesión activa: {st.session_state.datos_usuario.get('nombre','')}")
        if st.button("Cerrar sesión"):
            st.session_state.usuario_registrado = False
            st.session_state.datos_usuario = {}
            st.rerun()
        st.divider()

    st.title("🏗️ Sistema de Diseño de Micropilotes (MVP)")
    st.caption("Optimización simplificada + correlación SPT para estimación de qs. Para validación de mercado.")

    tab_diseno, tab_geo = st.tabs(["📐 Diseño & Optimización", "🌍 Correlación SPT → qs"])

    # -------- Tab 1: Diseño
    with tab_diseno:
        with st.sidebar:
            st.header("1) Estratigrafía")
            num_capas = st.slider("Número de capas", 1, 10, 3)

            ESTRATOS = []
            z_acum = 0.0

            with st.expander("Configurar capas", expanded=True):
                for i in range(num_capas):
                    st.markdown(f"**Capa {i+1}**")
                    c1, c2, c3 = st.columns(3)
                    esp = c1.number_input(f"Espesor (m) C{i+1}", value=3.0 if i == 0 else 5.0,
                                          step=0.5, key=f"esp_{i}")
                    qs = c2.number_input(f"qs (kPa) C{i+1}", value=40.0 + i * 20.0,
                                         step=5.0, key=f"qs_{i}")
                    fexp = c3.number_input(f"Factor exp. C{i+1}", value=1.1 + i * 0.05,
                                           step=0.1, min_value=1.0, max_value=3.0, key=f"fexp_{i}")
                    z_acum += esp
                    ESTRATOS.append({
                        "espesor": esp,
                        "z_fin": z_acum,
                        "qs": qs,
                        "f_exp": fexp,
                        "color": COLORES_ESTRATOS[i % len(COLORES_ESTRATOS)]
                    })
            st.caption(f"Profundidad total: {z_acum:.1f} m")

            st.header("2) Cargas y materiales")
            carga_ton = st.number_input("Carga total (Ton)", value=120.0, step=1.0)
            fs_req = st.slider("FS geotécnico", 1.5, 3.0, 2.0, 0.1)
            wc = st.slider("Relación A/C lechada", 0.40, 0.60, 0.50, 0.05)

            st.divider()
            calcular = st.button("Calcular diseño optimizado", type="primary")

        if calcular:
            df = optimizar_diseno(ESTRATOS, carga_ton, fs_req, wc)
            if df.empty:
                st.error("No se encontraron soluciones con los límites actuales (N, L, qs, etc.).")
                return

            best = df.iloc[0]

            st.subheader("Mejor opción (según Costo_Idx)")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Configuración", f"{int(best.N)} x Ø{int(best.D_mm)} mm")
            k2.metric("Longitud", f"{int(best.L_m)} m / micro", f"Perf total: {int(best.Perf_m)} m")
            k3.metric("Grout expandido", f"{best.Grout_m3:.1f} m³")
            k4.metric("CO2 estimado", f"{best.CO2_ton:.1f} ton")

            st.caption(f"Acero requerido aprox.: {best.Acero_req_cm2:.1f} cm² por micropilote (Fy=500MPa).")

            cA, cB = st.columns([1, 1.2])

            # Gráfica transferencia
            with cA:
                st.subheader("Transferencia de carga (admisible)")
                fig, ax = plt.subplots(figsize=(8, 6))

                y_lim = float(df["L_m"].max()) + 2.0
                prev_z = 0.0
                for e in ESTRATOS:
                    h = min(e["z_fin"], y_lim) - prev_z
                    if h > 0:
                        rect = patches.Rectangle((0, prev_z), 1e9, h, color=e["color"], alpha=0.25)
                        ax.add_patch(rect)
                        ax.text(0.5, prev_z + h/2,
                                f"qs={int(e['qs'])} kPa | fexp={e['f_exp']}",
                                va="center", fontsize=8, alpha=0.8)
                    prev_z = e["z_fin"]
                    if prev_z >= y_lim:
                        break

                # Curvas: dibujar 3-5 alternativas
                for i in range(min(5, len(df))):
                    row = df.iloc[i]
                    z_p, q_p = perfil_transferencia(row.D_val, row.L_m, ESTRATOS)
                    q_adm = [(q / fs_req) / 9.81 for q in q_p]  # Ton
                    ax.plot(q_adm, z_p, marker="o", linewidth=2, alpha=0.8,
                            label=f"Ø{int(row.D_mm)} | N={int(row.N)} | L={int(row.L_m)}m")

                ax.set_ylim(y_lim, 0)
                ax.set_xlabel("Capacidad admisible acumulada (Ton)")
                ax.set_ylabel("Profundidad (m)")
                ax.grid(True, linestyle=":", alpha=0.5)
                ax.legend(fontsize=8, loc="lower right")
                st.pyplot(fig)

            # Tabla + export
            with cB:
                st.subheader("Top alternativas")
                show = df[["D_mm","N","L_m","Perf_m","FS","Qadm_T","Qact_T","Grout_m3","CO2_ton","Costo_Idx"]].copy()
                show.columns = ["Ø(mm)","Cant","L(m)","Perf(m)","FS","Qadm(T)","Qact(T)","Grout(m³)","CO2(T)","CostoIdx"]
                st.dataframe(
                    show.head(12).style.format({
                        "FS":"{:.2f}","Qadm(T)":"{:.1f}","Qact(T)":"{:.1f}",
                        "Grout(m³)":"{:.1f}","CO2(T)":"{:.1f}","CostoIdx":"{:.0f}"
                    }),
                    use_container_width=True,
                    height=420
                )

                # Guardar run (trazabilidad)
                payload = {
                    "usuario": st.session_state.datos_usuario,
                    "inputs": {"carga_ton": carga_ton, "fs_req": fs_req, "wc": wc, "estratos": ESTRATOS},
                    "outputs": {"best": best.to_dict(), "top12": show.head(12).to_dict(orient="records")}
                }
                run_path = save_run(payload)

                # Descargas
                st.download_button(
                    "Descargar top alternativas (CSV)",
                    data=show.head(50).to_csv(index=False).encode("utf-8"),
                    file_name="resultados_micropilotes.csv",
                    mime="text/csv"
                )
                st.download_button(
                    "Descargar run (JSON trazabilidad)",
                    data=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
                    file_name="run_micropilotes.json",
                    mime="application/json"
                )
                st.caption(f"Run guardado en: {run_path.as_posix()}")

        else:
            st.info("Configure inputs en la barra lateral y ejecute el cálculo.")

    # -------- Tab 2: SPT -> qs
    with tab_geo:
        st.header("Correlación SPT → qs (MVP)")
        col1, col2 = st.columns([1, 1.2])

        with col1:
            st.markdown("Pegue datos: `Profundidad, N` (coma o tab).")
            spt_raw = st.text_area("Datos SPT", height=250,
                                   value="1.5, 4\n3.0, 7\n4.5, 12\n6.0, 15\n7.5, 22\n9.0, 28\n10.5, 35\n12.0, 42\n15.0, 50")
            k = st.slider("Factor K (qs ≈ K·N)", 1.0, 10.0, 3.5, 0.5)
            procesar = st.button("Procesar y graficar")

        with col2:
            st.markdown("""
**Uso previsto (MVP):** estimación preliminar de qs a partir de N-SPT.  
Se recomienda calibrar con ensayos de carga y/o parámetros de laboratorio cuando sea posible.
""")

        if procesar and spt_raw.strip():
            try:
                df_spt = pd.read_csv(StringIO(spt_raw), names=["z_m","N"], header=None,
                                     sep=r"[,\t]+", engine="python")
                df_spt["qs_kPa"] = df_spt["N"] * k

                fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
                axA.plot(df_spt["N"], df_spt["z_m"], marker="o")
                axA.set_title("Perfil N-SPT")
                axA.set_xlabel("N")
                axA.set_ylabel("Profundidad (m)")
                axA.grid(True, linestyle=":")

                axB.plot(df_spt["qs_kPa"], df_spt["z_m"], marker="s")
                axB.fill_betweenx(df_spt["z_m"], 0, df_spt["qs_kPa"], alpha=0.2)
                axB.set_title(f"qs estimado (K={k})")
                axB.set_xlabel("qs (kPa)")
                axB.grid(True, linestyle=":")

                axA.set_ylim(df_spt["z_m"].max() + 2, 0)
                st.pyplot(fig)

                st.dataframe(df_spt, use_container_width=True)

            except Exception as e:
                st.error(f"Error procesando SPT. Formato esperado: profundidad, N. Detalle: {e}")

# =========================
# Router
# =========================
if st.session_state.usuario_registrado:
    app_principal()
else:
    mostrar_registro()

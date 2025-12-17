import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re
import os
import io
import gc

# ==========================================
# 1. CONFIGURATION
# ==========================================
st.set_page_config(layout="wide", page_title="Wind Profile Analyzer")

REGEX_HEIGHT = re.compile(r'(?P<height>[0-9]+(?:\.[0-9]+)?)(?:m)?')

def get_sector_config(num_sectors: int):
    width = 360.0 / num_sectors
    start = -width / 2.0
    end = 360.0 - (width / 2.0)
    intervals = pd.interval_range(start=start, end=end, periods=num_sectors, closed='right')
    labels = [str(i) for i in range(num_sectors)]
    return intervals, labels

# ==========================================
# 2. MATHS : FIT LOG-LOG + RMSE
# ==========================================

def plot_wake_rose_streamlit(df, wd_col, waked_col, intervals, labels, title):
    df_plot = df.copy()
    width = 360 / len(labels)
    half_width = width / 2.0
    
    dirs = df_plot[wd_col] % 360
    dirs = dirs.mask(dirs > (360 - half_width), dirs - 360)
    
    df_plot['sect_code'] = pd.cut(dirs, bins=intervals, include_lowest=True).cat.codes
    df_valid = df_plot[df_plot['sect_code'] != -1].copy()
    
    if df_valid.empty:
        return go.Figure().update_layout(title="Pas de données valides")

    stats = df_valid.groupby('sect_code')[waked_col].agg(['count', 'sum']).reindex(range(len(labels)), fill_value=0)
    stats['waked'] = stats['sum']
    stats['free'] = stats['count'] - stats['sum']
    
    fig = go.Figure()
    fig.add_trace(go.Barpolar(r=stats['waked'], theta=labels, name='Affecté', marker_color='red'))
    fig.add_trace(go.Barpolar(r=stats['free'], theta=labels, name='Libre', marker_color='lightgrey'))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        font_size=11,
        legend=dict(orientation="h", y=-0.1),
        barmode='stack', 
        polar=dict(
            radialaxis=dict(showticklabels=True, angle=45, tickfont=dict(size=10)),
            angularaxis=dict(rotation=90, direction='clockwise', type='category', categoryorder='array', categoryarray=labels)
        ),
        margin=dict(l=40, r=40, t=60, b=40),
        height=450
    )
    return fig

def fit_power_law(heights, speeds):
    h = np.array(heights, dtype=float)
    v = np.array(speeds, dtype=float)
    mask = (h > 0) & (v > 0)
    h_clean, v_clean = h[mask], v[mask]
    
    if len(h_clean) < 2: return None, None, None 

    try:
        log_h, log_v = np.log(h_clean), np.log(v_clean)
        coeffs = np.polyfit(log_h, log_v, 1)
        alpha, beta = coeffs[0], coeffs[1]
        
        z_max = np.max(h_clean)
        z_smooth = np.linspace(0, z_max * 1.2, 100)
        v_smooth = np.exp(beta) * np.power(z_smooth, alpha)
        
        v_pred = np.exp(beta) * np.power(h_clean, alpha)
        rmse = np.sqrt(np.mean((v_clean - v_pred) ** 2))
        return alpha, rmse, (z_smooth, v_smooth)
    except:
        return None, None, None

# ==========================================
# 3. CHARGEMENT ROBUSTE
# ==========================================

def group_columns_by_sensor(df):
    sensors = {}
    for col in df.columns:
        c_str = str(col)
        if "Status" in c_str or "Code" in c_str: continue
        m = REGEX_HEIGHT.search(c_str)
        if not m: continue
        try:
            h = float(m.group('height'))
            if h.is_integer(): h = int(h)
        except: continue
        
        u_col = c_str.upper()
        ctype = "OTHER"
        if any(k in u_col for k in ["DIR", "WD", "DIRECTION"]): ctype = "WD"
        elif any(k in u_col for k in ["SPEED", "WS", "VITESSE", "MEAN", "AVG"]):
            if not any(k in u_col for k in ["MAX", "MIN", "STD", "TI", "TURB"]): ctype = "WS"
        if ctype == "OTHER": continue

        if h not in sensors: sensors[h] = {'WS': None, 'WD': None}
        if sensors[h][ctype] is None: sensors[h][ctype] = col
    return {h: d for h, d in sensors.items() if d['WS'] is not None}

def smart_load(input_source):
    if input_source is None: return None
    KEYWORDS = ['TimeStamp', 'Date & Time', 'MeanWindSpeed', 'TimeStampStatus']
    detected_header_row = 0
    encoding_used = 'utf-8'
    
    try:
        if isinstance(input_source, str):
            # Cas chemin local (ne devrait plus servir pour Clean, mais au cas où)
            if not os.path.exists(input_source): return None
            file_path = input_source
            if file_path.lower().endswith(('.xlsx', '.xls')):
                df_scan = pd.read_excel(file_path, header=None, nrows=50)
                for idx, row in df_scan.iterrows():
                    if any(k in " ".join(row.astype(str)) for k in KEYWORDS): detected_header_row = idx; break
                df = pd.read_excel(file_path, header=detected_header_row)
            else:
                try: 
                    with open(file_path, 'r', encoding='utf-8') as f: lines = [f.readline() for _ in range(60)]
                except:
                    encoding_used = 'ISO-8859-1'
                    with open(file_path, 'r', encoding='ISO-8859-1') as f: lines = [f.readline() for _ in range(60)]
                sep_candidate = '\t'
                for i, line in enumerate(lines):
                    if any(k in line for k in KEYWORDS):
                        detected_header_row = i
                        if ';' in line: sep_candidate = ';'
                        elif ',' in line: sep_candidate = ','
                        break
                try: df = pd.read_csv(file_path, header=0, skiprows=detected_header_row, sep=sep_candidate, encoding=encoding_used, on_bad_lines='skip', low_memory=False)
                except: df = pd.read_csv(file_path, header=0, skiprows=detected_header_row, sep=None, engine='python', encoding=encoding_used, on_bad_lines='skip')
        else:
            # Cas UploadedFile (Objet Streamlit)
            filename = input_source.name.lower()
            if filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(input_source)
            else:
                stringio = io.StringIO(input_source.getvalue().decode("ISO-8859-1"))
                lines = [stringio.readline() for _ in range(50)]
                sep_candidate = '\t'
                for i, line in enumerate(lines):
                    if any(k in line for k in KEYWORDS):
                        detected_header_row = i
                        if ';' in line: sep_candidate = ';'
                        elif ',' in line: sep_candidate = ','
                        break
                input_source.seek(0)
                df = pd.read_csv(input_source, header=0, skiprows=detected_header_row, sep=sep_candidate, encoding='ISO-8859-1', on_bad_lines='skip')
    except Exception as e:
        st.error(f"Erreur lecture : {e}")
        return None

    if df.empty: return None
    df.columns = [str(c).strip().replace('"', '').replace("'", "").split('|')[0] for c in df.columns]
    if not df.empty:
        row0 = str(df.iloc[0].values)
        if 'm/s' in row0 or 'Deg' in row0: df = df.iloc[1:].reset_index(drop=True)

    time_col = next((c for c in df.columns if 'TimeStamp' in c or ('Date' in c and 'Time' in c)), None)
    if time_col:
        df.rename(columns={time_col: 'TimeStamp'}, inplace=True)
        try: df['TimeStamp'] = pd.to_datetime(df['TimeStamp'], dayfirst=True, format='mixed', errors='coerce')
        except: df['TimeStamp'] = pd.to_datetime(df['TimeStamp'], dayfirst=True, errors='coerce')
        df.dropna(subset=['TimeStamp'], inplace=True)
        df['TimeStamp'] = df['TimeStamp'].dt.floor('min')
    else: return None

    for c in df.columns: 
        if c != 'TimeStamp': df[c] = pd.to_numeric(df[c], errors='coerce')
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    return df

# ==========================================
# 4. INTERFACE
# ==========================================

st.title("Profils de Vent : Comparateur Multi-Fichiers")

if 'data_raw' not in st.session_state: st.session_state['data_raw'] = None
if 'data_cod' not in st.session_state: st.session_state['data_cod'] = None
if 'clean_sources_map' not in st.session_state: st.session_state['clean_sources_map'] = {}
if 'clean_dfs_cache' not in st.session_state: st.session_state['clean_dfs_cache'] = {}

with st.expander("Configuration & Import", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 1. Fichier RAW")
        uploaded_raw = st.file_uploader("Fichier RAW", type=['txt', 'csv', 'xlsx', 'asc'], key="upl_raw")
    with c2:
        st.markdown("### 2. Dossier CLEAN")
        # CHANGEMENT ICI : Utilisation de accept_multiple_files=True au lieu de text_input
        uploaded_clean_files = st.file_uploader(
            "Sélectionner TOUS les fichiers CLEAN", 
            type=['txt', 'csv', 'xlsx', 'asc'], 
            accept_multiple_files=True,
            key="upl_clean"
        )
        if uploaded_clean_files:
            st.caption(f"{len(uploaded_clean_files)} fichiers sélectionnés.")
            
    with c3:
        st.markdown("### 3. Before COD (Opt)")
        uploaded_cod = st.file_uploader("Fichier Before COD", type=['txt', 'csv', 'xlsx', 'asc'], key="upl_cod")
    
    n_sect = st.number_input("Nombre de Secteurs", 4, 36, 12)
    intervals, labels = get_sector_config(n_sect)

    if st.button("CHARGER ET ANALYSER", type="primary"):
        st.session_state['clean_dfs_cache'] = {} 
        
        # 1. RAW
        if uploaded_raw:
            with st.spinner("Lecture RAW..."):
                st.session_state['data_raw'] = smart_load(uploaded_raw)
                if st.session_state['data_raw'] is not None: st.success("RAW OK.")
        else: st.error("Manque RAW.")
        
        # 2. CLEAN (Stockage des objets fichiers dans le dictionnaire)
        if uploaded_clean_files:
            # On stocke {nom_fichier: objet_fichier}
            st.session_state['clean_sources_map'] = {f.name: f for f in uploaded_clean_files}
            st.success(f"CLEAN : {len(uploaded_clean_files)} fichiers prêts.")
        else:
            st.warning("Aucun fichier CLEAN sélectionné.")
        
        # 3. BEFORE COD
        if uploaded_cod:
            with st.spinner("Lecture COD..."):
                st.session_state['data_cod'] = smart_load(uploaded_cod)
                if st.session_state['data_cod'] is not None: st.success("COD OK.")
        else: st.session_state['data_cod'] = None
        gc.collect()

# ==========================================
# 5. PLOT & ANALYSE
# ==========================================
if st.session_state['data_raw'] is not None and st.session_state['clean_sources_map']:
    df_raw = st.session_state['data_raw']
    df_cod = st.session_state['data_cod']
    # Liste des noms de fichiers disponibles
    available_sources = list(st.session_state['clean_sources_map'].keys())
    
    st.markdown("---")
    c_sel, c_dummy = st.columns([3, 1])
    selected_keys = c_sel.multiselect("2. Sélectionner fichiers CLEAN :", available_sources)
    
    if selected_keys:
        clean_dfs = {}
        for key in selected_keys:
            ckey = f"cache_{key}"
            if ckey not in st.session_state['clean_dfs_cache']:
                with st.spinner(f"Lecture {key}..."):
                    # On récupère l'objet fichier depuis la map
                    file_obj = st.session_state['clean_sources_map'][key]
                    st.session_state['clean_dfs_cache'][ckey] = smart_load(file_obj)
            if st.session_state['clean_dfs_cache'][ckey] is not None:
                clean_dfs[key] = st.session_state['clean_dfs_cache'][ckey]

        if clean_dfs:
            sensors_r = group_columns_by_sensor(df_raw)
            global_waked_sectors = set() 
            figures_ready = {}

            # --- CALCUL PRELIMINAIRE ---
            for name, df_c in clean_dfs.items():
                sensors_c = group_columns_by_sensor(df_c)
                m_check = pd.merge(df_raw[['TimeStamp']], df_c[['TimeStamp']], on='TimeStamp', how='inner')
                if m_check.empty: continue
                
                common_h = sorted(list(set(sensors_r.keys()).intersection(sensors_c.keys())), reverse=True)
                rose_data_per_height = {} 
                all_heights_dfs = [] 
                
                for h in common_h:
                    ws_r, wd_r = sensors_r[h]['WS'], sensors_r[h]['WD']
                    ws_c = sensors_c[h]['WS']
                    cur_wd = wd_r if wd_r else next((sensors_r[x]['WD'] for x in common_h if sensors_r[x]['WD']), None)
                    
                    if ws_r and cur_wd and ws_c:
                        m = pd.merge(df_raw[['TimeStamp', ws_r, cur_wd]], df_c[['TimeStamp', ws_c]], on='TimeStamp', how='inner')
                        if m.empty: continue
                        
                        diff = (m[ws_r] - m[ws_c]).abs()
                        mask_mod = (diff > 0.05) | (m[ws_r].notna() & m[ws_c].isna())
                        
                        width = 360 / n_sect
                        wd_vals_global = m.loc[mask_mod, cur_wd] % 360
                        wd_vals_global = wd_vals_global.mask(wd_vals_global > (360 - width/2), wd_vals_global - 360)
                        
                        if not wd_vals_global.empty:
                            cats = pd.cut(wd_vals_global, bins=intervals, include_lowest=True)
                            for c in cats.cat.codes.unique(): 
                                if c != -1: global_waked_sectors.add(int(c))
                        
                        df_rose = pd.DataFrame({'WD': m[cur_wd], 'is_waked': mask_mod})
                        rose_data_per_height[h] = df_rose
                        all_heights_dfs.append(df_rose)

                if all_heights_dfs:
                    rose_data_per_height['Global'] = pd.concat(all_heights_dfs, ignore_index=True)
                figures_ready[name] = {'heights': common_h, 'data': rose_data_per_height}

            waked_list = sorted(list(global_waked_sectors))
            
            c1, c2 = st.columns([1, 3])
            
            with c1:
                st.markdown("### 1. Analyse (Clean)")
                st.info(f"Secteurs Modifiés : {waked_list if waked_list else 'Aucun'}")
                all_opts = [f"{i} ({labels[i]})" for i in range(n_sect)]
                
                sel = st.multiselect("Secteurs 'Analysés'", all_opts, default=[all_opts[i] for i in waked_list] if waked_list else [])
                sel_idx = [int(s.split(' ')[0]) for s in sel]
                use_only_mod = st.checkbox("Filtrer : Uniquement modifiés", value=True)
                
                st.markdown("---")
                st.markdown("### 2. Référence")
                if df_cod is not None:
                    ref_options = ["RAW (Actuel)", "BEFORE COD (Historique)"]
                else:
                    ref_options = ["RAW (Actuel)"]

                ref_source = st.radio("Source Référence :", ref_options)
                
                # --- DOUBLE FILTRE ---
                st.caption("Filtres Référence :")
                sel_sect_ref = st.multiselect("Secteurs à inclure", all_opts, default=all_opts)
                sel_idx_ref = [int(s.split(' ')[0]) for s in sel_sect_ref]
                
                ref_point_filter = st.radio(
                    "Type de points", 
                    ["Tout", "Points Free (Non-affectés)", "Points Affectés (Waked)"],
                    horizontal=False
                )

                st.markdown("---")
                available_heights = sorted(list(sensors_r.keys()), reverse=True)
                sel_heights = st.multiselect("Hauteurs Graphique", available_heights, default=available_heights)
                
                if figures_ready:
                    st.markdown("### Roses")
                    for fname, content in figures_ready.items():
                        with st.expander(f"Rose: {fname}", expanded=False):
                            opts = ['Global'] + sorted(content['heights'], reverse=True)
                            choice = st.selectbox(f"H ({fname})", opts, key=f"sel_{fname}")
                            if choice in content['data']:
                                fig_rose = plot_wake_rose_streamlit(content['data'][choice], 'WD', 'is_waked', intervals, labels, title=f"{fname} @{choice}")
                                st.plotly_chart(fig_rose, width='stretch')

            with c2:
                if sel_heights:
                    fig = go.Figure()
                    colors = ['#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880']
                    
                    # =========================================================================
                    # 1. TRACÉ REF
                    # =========================================================================
                    hv_ref, pr_ref = [], []
                    ref_group_name = "REF_TRACE" 

                    if "BEFORE COD" in ref_source and df_cod is not None:
                        # --- BEFORE COD ---
                        sensors_cod = group_columns_by_sensor(df_cod)
                        for h in sel_heights:
                            if h in sensors_cod:
                                ws_cod, wd_cod = sensors_cod[h]['WS'], sensors_cod[h]['WD']
                                if not wd_cod:
                                    for ha in sensors_cod: 
                                        if sensors_cod[ha]['WD']: wd_cod = sensors_cod[ha]['WD']; break
                                
                                if ws_cod and wd_cod:
                                    d_cod = df_cod[[ws_cod, wd_cod]].copy()
                                    d_cod.dropna(inplace=True)
                                    width = 360 / n_sect
                                    wd_vals = d_cod[wd_cod] % 360
                                    wd_vals = wd_vals.mask(wd_vals > (360 - width/2), wd_vals - 360)
                                    sect_codes = pd.cut(wd_vals, bins=intervals, include_lowest=True).cat.codes
                                    
                                    d_filtered = d_cod[sect_codes.isin(sel_idx_ref)]
                                    if not d_filtered.empty:
                                        hv_ref.append(h)
                                        pr_ref.append(d_filtered[ws_cod].mean())
                        label_ref_name = "Before COD"
                    
                    else:
                        # --- RAW ACTUEL ---
                        for h in sel_heights:
                            if h in sensors_r:
                                ws_col, wd_col = sensors_r[h]['WS'], sensors_r[h]['WD']
                                if not wd_col:
                                    for ha in sensors_r: 
                                        if sensors_r[ha]['WD']: wd_col = sensors_r[ha]['WD']; break
                                if ws_col and wd_col:
                                    d_raw_h = df_raw[['TimeStamp', ws_col, wd_col]].dropna()
                                    width = 360 / n_sect
                                    wd_vals = d_raw_h[wd_col] % 360
                                    wd_vals = wd_vals.mask(wd_vals > (360 - width/2), wd_vals - 360)
                                    sect_codes = pd.cut(wd_vals, bins=intervals, include_lowest=True).cat.codes
                                    
                                    d_ref = d_raw_h[sect_codes.isin(sel_idx_ref)].copy()
                                    
                                    if not d_ref.empty and ref_point_filter != "Tout":
                                        is_waked_mask = pd.Series(False, index=d_ref.index)
                                        for _, df_c in clean_dfs.items():
                                            sensors_c = group_columns_by_sensor(df_c)
                                            if h in sensors_c:
                                                ws_c = sensors_c[h]['WS']
                                                m_check = pd.merge(d_ref[['TimeStamp', ws_col]], df_c[['TimeStamp', ws_c]], on='TimeStamp', how='left')
                                                m_check['diff'] = (m_check[ws_col] - m_check[ws_c]).abs()
                                                waked_in_this_file = (m_check['diff'] > 0.05) | (m_check[ws_col].notna() & m_check[ws_c].isna())
                                                waked_in_this_file.index = d_ref.index
                                                is_waked_mask = is_waked_mask | waked_in_this_file

                                        if "Free" in ref_point_filter:
                                            d_ref = d_ref[~is_waked_mask]
                                        else: # Affectés
                                            d_ref = d_ref[is_waked_mask]

                                    if not d_ref.empty:
                                        hv_ref.append(h)
                                        pr_ref.append(d_ref[ws_col].mean())
                        label_ref_name = "Raw Ref"

                    # PLOT REF
                    if len(hv_ref) >= 2:
                        alpha_ref, rmse_ref, smooth_ref = fit_power_law(hv_ref, pr_ref)
                        lbl = f"{label_ref_name}"
                        if alpha_ref: lbl += f" [α={alpha_ref:.2f}, RMSE={rmse_ref:.2f}]"
                        
                        if smooth_ref:
                            fig.add_trace(go.Scatter(x=smooth_ref[1], y=smooth_ref[0], mode='lines', 
                                                     line=dict(color='black', dash='dash', width=3), 
                                                     name=lbl, legendgroup=ref_group_name))
                        fig.add_trace(go.Scatter(x=pr_ref, y=hv_ref, mode='markers', 
                                                 marker=dict(color='black', symbol='x', size=8), 
                                                 showlegend=False, legendgroup=ref_group_name))

                    # =========================================================================
                    # 2. TRACÉ CLEAN
                    # =========================================================================
                    if sel_idx:
                        max_x_seen = 0
                        if pr_ref: max_x_seen = max(pr_ref)

                        for idx_item, (name, df_c) in enumerate(clean_dfs.items()):
                            sensors_c = group_columns_by_sensor(df_c)
                            intersect_h = set(sensors_r.keys()).intersection(sensors_c.keys())
                            comm = sorted([h for h in intersect_h if h in sel_heights], reverse=True)
                            
                            hv, pc = [], []
                            for h in comm:
                                ws_r, wd_r = sensors_r[h]['WS'], sensors_r[h]['WD']
                                ws_c = sensors_c[h]['WS']
                                if not wd_r: 
                                    for ha in comm: 
                                        if sensors_r[ha]['WD']: wd_r = sensors_r[ha]['WD']; break
                                if ws_r and wd_r and ws_c:
                                    m = pd.merge(df_raw[['TimeStamp', wd_r, ws_r]], df_c[['TimeStamp', ws_c]], on='TimeStamp', how='inner')
                                    if m.empty: continue
                                    width = 360 / n_sect
                                    wd_vals = m[wd_r] % 360
                                    wd_vals = wd_vals.mask(wd_vals > (360 - width/2), wd_vals - 360)
                                    m['sect'] = pd.cut(wd_vals, bins=intervals, include_lowest=True).cat.codes
                                    d = m[m['sect'].isin(sel_idx)]
                                    if use_only_mod and not d.empty:
                                        diff = (d[ws_r] - d[ws_c]).abs()
                                        mask_diff = (diff > 0.05) | (d[ws_r].notna() & d[ws_c].isna())
                                        d = d[mask_diff]
                                    if not d.empty:
                                        hv.append(h)
                                        pc.append(d[ws_c].mean())
                            
                            if pc: max_x_seen = max(max_x_seen, max(pc))
                            color = colors[idx_item % len(colors)]
                            
                            if len(hv) >= 2:
                                alpha, rmse, smooth = fit_power_law(hv, pc)
                                label_clean = f"{name}"
                                if alpha: label_clean += f" (α={alpha:.2f}, RMSE={rmse:.2f})"
                                
                                if smooth:
                                    fig.add_trace(go.Scatter(x=smooth[1], y=smooth[0], mode='lines', 
                                                             line=dict(color=color, width=2), 
                                                             name=label_clean, legendgroup=name))
                                fig.add_trace(go.Scatter(x=pc, y=hv, mode='markers', 
                                                         marker=dict(color=color, size=6), 
                                                         showlegend=False, legendgroup=name))

                        limit_x = max_x_seen * 1.2 if max_x_seen > 0 else 25
                        fig.update_layout(title="Profils Verticaux", xaxis_title="Vitesse (m/s)", yaxis_title="Hauteur (m)", height=750, xaxis=dict(range=[0, limit_x]))
                        st.plotly_chart(fig, width='stretch')
                    else: st.warning("Sélectionnez au moins un secteur d'analyse.")

            # ==========================================
            # 6. TABLEAU COMPARATIF
            # ==========================================
            st.markdown("---")
            st.header("3. Tableau Comparatif (Synthèse)")
            
            calc_mode = st.radio("Base calcul :", ["Points modifiés", "Tous points"], horizontal=True)
            global_results = []
            
            for name_t, df_clean_t in clean_dfs.items():
                sensors_c_t = group_columns_by_sensor(df_clean_t)
                common_h_t = sorted(list(set(sensors_r.keys()).intersection(sensors_c_t.keys())), reverse=True)
                file_raw_means, file_clean_means, file_deltas, total_pts = [], [], [], 0
                
                for h_t in common_h_t:
                    if h_t not in sel_heights: continue

                    ws_r_t = sensors_r[h_t]['WS']
                    ws_c_t = sensors_c_t[h_t]['WS']
                    m_t = pd.merge(df_raw[['TimeStamp', ws_r_t]], df_clean_t[['TimeStamp', ws_c_t]], on='TimeStamp', how='inner')
                    if m_t.empty: continue
                    
                    diff_t = (m_t[ws_r_t] - m_t[ws_c_t]).abs()
                    mask_mod_t = (diff_t > 0.05) | (m_t[ws_r_t].notna() & m_t[ws_c_t].isna())
                    df_final = m_t[mask_mod_t] if calc_mode == "Points modifiés" else m_t
                    
                    if not df_final.empty:
                        mean_raw = df_final[ws_r_t].mean()
                        mean_clean = df_final[ws_c_t].mean()
                        delta = ((mean_clean - mean_raw) / mean_raw) * 100 if mean_raw else 0.0
                        file_raw_means.append(mean_raw)
                        file_clean_means.append(mean_clean)
                        file_deltas.append(delta)
                        total_pts += len(df_final)

                if file_raw_means:
                    avg_raw = sum(file_raw_means) / len(file_raw_means)
                    avg_clean = sum(file_clean_means) / len(file_clean_means)
                    avg_delta = sum(file_deltas) / len(file_deltas)
                    global_results.append({
                        "Fichier": name_t,
                        "Nb Hauteurs": len(file_raw_means),
                        "Pts (Cumul)": total_pts,
                        "Moy RAW": avg_raw,
                        "Moy CLEAN": avg_clean,
                        "Delta (%)": avg_delta
                    })
            
            if global_results:
                df_stats = pd.DataFrame(global_results)
                st.dataframe(
                    df_stats.style.format({"Moy RAW": "{:.3f}", "Moy CLEAN": "{:.3f}", "Delta (%)": "{:+.2f}%"}).background_gradient(subset=['Delta (%)'], cmap='RdYlGn', vmin=-5, vmax=5),
                    use_container_width=True,
                    height=len(global_results) * 35 + 38
                )


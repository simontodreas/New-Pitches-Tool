import sys, os
sys.path.insert(0, '/Users/kids/Pitcher Similarity')

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd

from pitch_suggestions import suggest_pitches

SNAPSHOT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'snapshots')
SNAPSHOT_KEYS = ['pitcher_summ_r', 'pitcher_summ_l', 'pitch_type_r', 'pitch_type_l']

st.set_page_config(page_title="Pitch Suggestions", layout="wide")

st.markdown("""
<style>
.metric-label { font-size: 0.8rem; color: #888; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_data():
    """Prefer the prebuilt Parquet snapshot; fall back to building from CSVs."""
    if all(os.path.exists(os.path.join(SNAPSHOT_DIR, f'{k}.parquet')) for k in SNAPSHOT_KEYS):
        return {k: pd.read_parquet(os.path.join(SNAPSHOT_DIR, f'{k}.parquet')) for k in SNAPSHOT_KEYS}
    from data import build_all  # heavy path: only imported if the snapshot is missing
    data = build_all(live=False)
    return {k: data[k] for k in SNAPSHOT_KEYS}


@st.cache_data(show_spinner=False)
def run_suggest(pitcher_name, is_righty, biomech_thr, novelty_thr, min_usage, min_pitches):
    data = load_data()
    pitcher_summ   = data['pitcher_summ_r']  if is_righty else data['pitcher_summ_l']
    pitch_type_summ = data['pitch_type_r']   if is_righty else data['pitch_type_l']
    return suggest_pitches(
        target_pitcher=pitcher_name,
        pitcher_summ=pitcher_summ,
        pitch_type_summ=pitch_type_summ,
        biomech_distance_threshold=biomech_thr,
        novelty_distance_threshold=novelty_thr,
        min_comp_usage_pct=min_usage,
        min_pitches=min_pitches,
    )


def make_cluster_fig(result, is_righty):
    comp_pitches   = result['comp_pitches']
    target_pitches = result['target_pitches']
    pitcher_name   = target_pitches['player_name'].iloc[0]

    plotly_markers = ['circle', 'square', 'triangle-up', 'diamond', 'cross',
                      'x', 'triangle-down', 'triangle-left', 'triangle-right', 'hexagon']
    cluster_keys = sorted(
        comp_pitches[['cluster_label', 'cluster']].drop_duplicates().itertuples(index=False, name=None)
    )
    cluster_key_index = {(label, cid): idx for idx, (label, cid) in enumerate(cluster_keys)}

    arm_angle_deg  = result['target_info']['arm_angle']
    arm_angle_rad  = np.radians(arm_angle_deg)

    vmin = comp_pitches['release_speed'].min()
    vmax = comp_pitches['release_speed'].max()

    fig = go.Figure()

    for i, (label, cid) in enumerate(cluster_keys):
        grp = comp_pitches[(comp_pitches['cluster_label'] == label) & (comp_pitches['cluster'] == cid)]
        fig.add_trace(go.Scatter(
            x=grp['pfx_x'],
            y=grp['pfx_z'],
            mode='markers',
            name=f'Possible Pitch ({label})',
            marker=dict(
                symbol=plotly_markers[i % len(plotly_markers)],
                size=8,
                color=grp['release_speed'],
                colorscale='plasma',
                cmin=vmin,
                cmax=vmax,
                opacity=0.7,
                showscale=(i == 0),
                colorbar=dict(
                    title=dict(text='Release Speed (mph)', side='right'),
                    x=1.02,
                    thickness=15,
                    len=0.75,
                ) if i == 0 else None,
            ),
            customdata=grp[['player_name', 'pitch_type']].values,
            hovertemplate=(
                '<b>%{customdata[0]}</b><br>'
                'Pitch: %{customdata[1]}'
                '<extra></extra>'
            ),
        ))

    centroids = comp_pitches.groupby(['cluster_label', 'cluster'])[['pfx_x', 'pfx_z', 'release_speed']].mean().reset_index()
    for _, row in centroids.iterrows():
        idx = cluster_key_index.get((row['cluster_label'], row['cluster']), 0)
        label = row['cluster_label']
        fig.add_trace(go.Scatter(
            x=[row['pfx_x']],
            y=[row['pfx_z']],
            mode='markers',
            name='Cluster Centroid',
            showlegend=(idx == 0),
            legendgroup='centroid',
            marker=dict(
                symbol=plotly_markers[idx % len(plotly_markers)],
                size=16,
                color=[row['release_speed']],
                colorscale='plasma',
                cmin=vmin,
                cmax=vmax,
                line=dict(color='black', width=2),
                showscale=False,
            ),
            hovertemplate=(
                f'<b>Centroid: {label}</b><br>'
                'HBreak: %{x:.2f} ft<br>'
                'IVBreak: %{y:.2f} ft'
                '<extra></extra>'
            ),
        ))

    if target_pitches is not None and not target_pitches.empty:
        fig.add_trace(go.Scatter(
            x=target_pitches['pfx_x'],
            y=target_pitches['pfx_z'],
            mode='markers+text',
            name='Existing Pitch',
            marker=dict(symbol='diamond', size=16, color='black'),
            text=target_pitches['pitch_type'],
            textposition='top right',
            textfont=dict(size=14, color='black'),
            customdata=target_pitches[['player_name', 'pitch_type']].values,
            hovertemplate=(
                '<b>%{customdata[0]}</b><br>'
                'Pitch: %{customdata[1]}'
                '<extra></extra>'
            ),
        ))

    # ── Arm angle (drawn on the main plot, pivoting at the origin) ─────────────
    ARM_LEN = 1.5
    arm_dir = -1 if is_righty else 1  # mirror righties so the arm enters from the correct side
    ax_x = arm_dir * ARM_LEN * np.cos(arm_angle_rad)
    ax_y = ARM_LEN * np.sin(arm_angle_rad)

    fig.add_trace(go.Scatter(
        x=[0, ax_x], y=[0, ax_y], mode='lines',
        name='Arm Angle',
        line=dict(color='rgba(50,50,50,0.30)', width=6),
        hovertemplate=f'Arm Angle: {arm_angle_deg:.1f}°<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=[ax_x], y=[ax_y], mode='markers',
        showlegend=False, legendgroup='Arm Angle',
        marker=dict(size=20, color='rgba(80,80,80,0.30)',
                    line=dict(color='rgba(50,50,50,0.35)', width=2)),
        hovertemplate=f'Arm Angle: {arm_angle_deg:.1f}°<extra></extra>',
    ))

    axis_range = [-2.5, 2.5]
    grid_style = dict(
        showgrid=True, gridcolor='lightgrey', gridwidth=1,
        zeroline=True, zerolinecolor='darkgrey', zerolinewidth=1.5,
        range=axis_range, constrain='domain',
    )
    fig.update_layout(
        title=dict(text=f'Potential Arsenal — {pitcher_name}', x=0.5, xanchor='center'),
        xaxis_title='Horizontal Break (ft)',
        yaxis_title='Induced Vertical Break (ft)',
        xaxis=grid_style,
        yaxis=dict(**grid_style, scaleanchor='x', scaleratio=1),
        dragmode=False,
        legend=dict(x=1.22, y=1, xanchor='left'),
        height=560,
        margin=dict(r=200),
    )
    return fig


# ── Load data ────────────────────────────────────────────────────────────────
with st.spinner("Loading data..."):
    data = load_data()

pitcher_summ_r = data['pitcher_summ_r']
pitcher_summ_l = data['pitcher_summ_l']

pitchers_r = set(pitcher_summ_r[pitcher_summ_r['game_year'] == 2025]['player_name'].unique())
pitchers_l = set(pitcher_summ_l[pitcher_summ_l['game_year'] == 2025]['player_name'].unique())
all_pitchers = sorted(pitchers_r | pitchers_l)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("Pitch Suggestions")

search = st.sidebar.text_input("Search pitcher", placeholder="e.g. Bello")
filtered = [p for p in all_pitchers if search.lower() in p.lower()] if search else all_pitchers

if not filtered:
    st.sidebar.warning("No pitchers match that search.")
    st.stop()

selected = st.sidebar.selectbox("Select pitcher", filtered)

st.sidebar.markdown("---")
st.sidebar.subheader("Parameters")

biomech_thr  = st.sidebar.slider("Biomech Distance Threshold",  0.5, 3.0, 1.5, 0.1,
                                  help="Max biomechanical distance to qualify as a comp")
novelty_thr  = st.sidebar.slider("Novelty Distance Threshold",  0.5, 3.0, 1.2, 0.1,
                                  help="Min pitch-char distance to count as novel vs. target")
min_usage    = st.sidebar.slider("Min Comp Usage %",            0.01, 0.10, 0.01, 0.01,
                                  help="Minimum usage share a comp pitch must have")
min_pitches  = st.sidebar.slider("Min Pitches",                 10, 50, 20, 5,
                                  help="Minimum pitch count to include a pitcher")

# ── Handedness ────────────────────────────────────────────────────────────────
is_righty = selected in pitchers_r

# ── Main ──────────────────────────────────────────────────────────────────────
throws = "RHP" if is_righty else "LHP"
st.title(f"Pitch Suggestions — {selected}")
st.caption(f"{throws}")

with st.spinner("Running analysis..."):
    result = run_suggest(selected, is_righty, biomech_thr, novelty_thr, min_usage, min_pitches)

status = result['status']

STATUS_MESSAGES = {
    'pitcher_not_found': "Pitcher not found in the dataset.",
    'no_comps':          "No biomechanically similar comps found. Try raising the Biomech Distance Threshold.",
    'no_comp_pitches':   "Comps found but no usable pitch data. Try lowering Min Comp Usage % or Min Pitches.",
    'no_novel_pitches':  "No novel pitches found. Try raising the Novelty Distance Threshold.",
}

if status != 'ok':
    st.warning(STATUS_MESSAGES.get(status, f"Status: {status}"))

    if result.get('comps') is not None and not result['comps'].empty:
        with st.expander("Similar Pitchers Found"):
            st.dataframe(result['comps'], use_container_width=True)
    st.stop()

# ── Target info metrics ───────────────────────────────────────────────────────
info = result['target_info']
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Arm Angle",  f"{info['arm_angle']:.1f}°")
c2.metric("Extension",  f"{info['release_extension']:.2f} ft")
c3.metric("Max Velo",   f"{info['max_velo']:.1f} mph")
c4.metric("Primary FB", info.get('pri_fb', 'N/A'))
c5.metric("Comps Found", len(result['comps']))

st.markdown("---")

# ── Current arsenal ───────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 1.6])

with left_col:
    st.subheader("Arsenal & Suggestions")

    target = result['target_pitches'].copy()
    total_n = target['n'].sum()

    current_rows = pd.DataFrame({
        'Pitch':                  target['pitch_type'].values,
        'Usage':                  (target['n'] / total_n).values,
        'Velocity':               target['release_speed'].values,
        'Horizontal Break':       target['pfx_x'].values,
        'Induced Vertical Break': target['pfx_z'].values,
        '# Comps':                np.nan,
    })

    sugg = result['suggestions']
    sugg_rows = pd.DataFrame({
        'Pitch':                  sugg['cluster_label'].values,
        'Usage':                  np.nan,
        'Velocity':               sugg['wavg_release_speed'].values,
        'Horizontal Break':       sugg['wavg_pfx_x'].values,
        'Induced Vertical Break': sugg['wavg_pfx_z'].values,
        '# Comps':                sugg['n_comps'].values.astype(float),
    })

    combined = pd.concat([current_rows, sugg_rows], ignore_index=True)

    st.dataframe(
        combined.style.format({
            'Usage':                  '{:.1%}',
            'Velocity':               '{:.1f}',
            'Horizontal Break':       '{:.2f}',
            'Induced Vertical Break': '{:.2f}',
            '# Comps':                '{:.0f}',
        }, na_rep=''),
        use_container_width=True,
        hide_index=True,
    )

with right_col:
    st.subheader("Cluster Plot")
    fig = make_cluster_fig(result, is_righty)
    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': False, 'displayModeBar': False})

st.markdown("---")

# ── Detail tables ─────────────────────────────────────────────────────────────
with st.expander("Comp Pitchers"):
    st.dataframe(result['comps'], use_container_width=True, hide_index=True)

with st.expander("Novel Comp Pitches"):
    cp = result['comp_pitches'].copy()
    display_cols = ['player_name', 'game_year', 'pitch_type', 'release_speed',
                    'pfx_x', 'pfx_z', 'usage_pct', 'min_dist_to_target',
                    'closest_target_pitch', 'cluster_label', 'biomech_distance']
    display_cols = [c for c in display_cols if c in cp.columns]
    st.dataframe(cp[display_cols], use_container_width=True, hide_index=True)


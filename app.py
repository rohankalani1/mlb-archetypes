import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

st.set_page_config(
    page_title="MLB Batter Archetype Classifier",
    page_icon="⚾",
    layout="wide"
)

ARCHETYPE_COLORS = {
    'Balanced Hitter':    '#F97316',
    'Contact Specialist': '#06B6D4',
    'Elite Power Hitter': '#A855F7',
    'Aggressive Hacker':  '#10B981',
}

ARCHETYPE_DESCRIPTIONS = {
    'Balanced Hitter':    'Solid BB%, good barrel rate, selective approach. Well-rounded producer.',
    'Contact Specialist': 'Low whiff%, high in-zone contact, rarely chases. Puts the ball in play.',
    'Elite Power Hitter': 'Elite barrel rate and walk rate. The best of both worlds.',
    'Aggressive Hacker':  'High chase rate, high whiff%, above-avg power. Lives by the swing.',
}

archetype_map = {
    0: 'Balanced Hitter',
    1: 'Contact Specialist',
    2: 'Elite Power Hitter',
    3: 'Aggressive Hacker',
}

chosen_features = ['BB%', 'Barrel Rate', 'In Zone Swing %', 'Out of Zone Swing %', 'Whiff %']

def _ordinal(n):
    n = int(round(n))
    suffix = 'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f"{n}{suffix}"
all_features = [
    'K%', 'BB%', 'ISO (SLG-AVG)', 'xwoba', 'Avg EV', 'Barrel Rate',
    'Hard Hit %', 'In Zone Swing %', 'Out of Zone Swing %',
    'Out of Zone Contact %', 'In Zone Contact %', 'Whiff %', 'Swing %',
    'Pull %', 'First Pitch Strike %', 'Ground Ball %', 'Fly Ball %',
    'Line Drive %', 'Speed (ft/sec)'
]

@st.cache_resource
def build_model():
    df = pd.read_csv('stats2.csv')
    df = df.rename(columns={
        'last_name, first_name': 'full name',
        'k_percent': 'K%', 'bb_percent': 'BB%',
        'isolated_power': 'ISO (SLG-AVG)', 'exit_velocity_avg': 'Avg EV',
        'barrel_batted_rate': 'Barrel Rate', 'hard_hit_percent': 'Hard Hit %',
        'z_swing_percent': 'In Zone Swing %', 'oz_swing_percent': 'Out of Zone Swing %',
        'oz_contact_percent': 'Out of Zone Contact %', 'iz_contact_percent': 'In Zone Contact %',
        'whiff_percent': 'Whiff %', 'swing_percent': 'Swing %',
        'pull_percent': 'Pull %', 'f_strike_percent': 'First Pitch Strike %',
        'groundballs_percent': 'Ground Ball %', 'flyballs_percent': 'Fly Ball %',
        'linedrives_percent': 'Line Drive %', 'sprint_speed': 'Speed (ft/sec)',
    })

    def weighted_avg(group):
        w = group['pa']
        return pd.Series({f: (group[f] * w).sum() / w.sum() for f in all_features})

    df_agg = df.groupby('player_id').apply(weighted_avg, include_groups=False).round(3)
    df_meta = df.groupby('player_id').agg(
        player_name=('full name', 'last'),
        total_pa=('pa', 'sum'),
        seasons_played=('year', 'count'),
    ).reset_index()
    df_agg = df_agg.reset_index().merge(df_meta, on='player_id')
    df_agg = df_agg[(df_agg['total_pa'] >= 100) & (df_agg['seasons_played'] >= 3)]

    df_model = df_agg[['player_id', 'player_name'] + chosen_features].dropna().copy()
    X = df_model[chosen_features]

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('km', KMeans(n_clusters=4, random_state=42, n_init=10))
    ])
    pipeline.fit(X)
    df_model['Cluster'] = pipeline.predict(X)
    df_model['Archetype'] = df_model['Cluster'].map(archetype_map)

    X_sc = pipeline['scaler'].transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_sc)
    df_model['PCA1'] = X_pca[:, 0]
    df_model['PCA2'] = X_pca[:, 1]

    return pipeline, pca, df_model, X_sc

pipeline, pca, df_model, X_sc = build_model()

_defaults = {
    'inp_name': '', 'inp_bb': 8.5, 'inp_barrel': 10.2,
    'inp_iz': 68.4, 'inp_oz': 28.1, 'inp_whiff': 24.3,
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

def _on_lookup():
    sel = st.session_state.get('player_lookup', '')
    if sel and sel != '— select a player —':
        row = df_model[df_model['player_name'] == sel].iloc[0]
        st.session_state['inp_bb']     = float(row['BB%'])
        st.session_state['inp_barrel'] = float(row['Barrel Rate'])
        st.session_state['inp_iz']     = float(row['In Zone Swing %'])
        st.session_state['inp_oz']     = float(row['Out of Zone Swing %'])
        st.session_state['inp_whiff']  = float(row['Whiff %'])
        parts = sel.split(', ')
        st.session_state['inp_name'] = f"{parts[1]} {parts[0]}" if len(parts) == 2 else sel

# ── Header ────────────────────────────────────────────────────────────────────
st.title("⚾ MLB Batter Archetype Classifier")
st.caption("K-Means clustering on 2021–2025 Statcast data  •  106 qualified players  •  4 archetypes")
st.divider()

# ── Layout ────────────────────────────────────────────────────────────────────
left, right = st.columns([1, 2.2])

with left:
    st.subheader("Player Input")

    lookup_options = ['— select a player —'] + sorted(df_model['player_name'].tolist())
    st.selectbox("Look up existing player", lookup_options, key='player_lookup', on_change=_on_lookup)
    st.caption("Selecting a player auto-fills their stats below.")

    player_name = st.text_input("Player Name", placeholder="e.g. Shohei Ohtani", key='inp_name')

    st.markdown("**Batting Stats**")
    bb       = st.number_input("BB%",         min_value=0.0,  max_value=30.0, step=0.1, key='inp_bb',     help="Walk rate")
    barrel   = st.number_input("Barrel Rate", min_value=0.0,  max_value=30.0, step=0.1, key='inp_barrel', help="Barrel batted ball rate")
    iz_swing = st.number_input("IZ Swing %",  min_value=40.0, max_value=90.0, step=0.1, key='inp_iz',     help="In-zone swing rate")
    oz_swing = st.number_input("OZ Swing %",  min_value=5.0,  max_value=55.0, step=0.1, key='inp_oz',     help="Out-of-zone chase rate")
    whiff    = st.number_input("Whiff %",     min_value=5.0,  max_value=50.0, step=0.1, key='inp_whiff',  help="Miss rate on all swings")

    classify = st.button("Classify Player", type="primary", use_container_width=True)

    st.divider()
    st.markdown("**Archetype Key**")
    for arch, color in ARCHETYPE_COLORS.items():
        st.markdown(
            f'<span style="color:{color}; font-weight:600;">■</span> '
            f'<span style="font-size:13px;">{arch}</span>',
            unsafe_allow_html=True
        )

with right:
    st.subheader("Archetype Map — PCA Projection")

    new_player_pca  = None
    new_archetype   = None
    similar_players = []

    if classify and player_name.strip():
        new_player = pd.DataFrame([{
            'BB%': bb, 'Barrel Rate': barrel,
            'In Zone Swing %': iz_swing, 'Out of Zone Swing %': oz_swing,
            'Whiff %': whiff
        }])
        cluster_id    = pipeline.predict(new_player)[0]
        new_archetype = archetype_map[cluster_id]
        new_sc        = pipeline['scaler'].transform(new_player)
        new_player_pca = pca.transform(new_sc)

        cluster_mask = df_model['Cluster'].values == cluster_id
        dists        = cdist(new_sc, X_sc[cluster_mask], metric='euclidean')[0]
        sorted_idx   = dists.argsort()
        start        = 1 if dists[sorted_idx[0]] < 0.001 else 0
        similar_players = [
            df_model[cluster_mask].iloc[sorted_idx[i]]['player_name']
            for i in range(start, start + 3)
        ]

    elif classify and not player_name.strip():
        st.warning("Enter a player name before classifying.")

    # Draw map
    fig = go.Figure()

    for arch, color in ARCHETYPE_COLORS.items():
        m = df_model['Archetype'] == arch
        subset = df_model[m]
        fig.add_trace(go.Scatter(
            x=subset['PCA1'],
            y=subset['PCA2'],
            mode='markers',
            name=arch,
            marker=dict(color=color, size=8, opacity=0.75),
            customdata=subset[['player_name', 'BB%', 'Barrel Rate',
                                'In Zone Swing %', 'Out of Zone Swing %', 'Whiff %']].values,
            hovertemplate=(
                '<b>%{customdata[0]}</b><br>'
                'BB%%: %{customdata[1]:.1f}  |  Barrel: %{customdata[2]:.1f}<br>'
                'IZ Swing: %{customdata[3]:.1f}  |  OZ Swing: %{customdata[4]:.1f}<br>'
                'Whiff: %{customdata[5]:.1f}'
                '<extra></extra>'
            ),
        ))

    looked_up = st.session_state.get('player_lookup', '— select a player —')
    if looked_up and looked_up != '— select a player —':
        lu = df_model[df_model['player_name'] == looked_up]
        if not lu.empty:
            lu = lu.iloc[0]
            fig.add_trace(go.Scatter(
                x=[lu['PCA1']],
                y=[lu['PCA2']],
                mode='markers+text',
                name=looked_up,
                marker=dict(
                    color=ARCHETYPE_COLORS[lu['Archetype']],
                    size=15,
                    line=dict(color='white', width=2),
                ),
                text=[lu['player_name'].split(', ')[0]],
                textposition='top right',
                textfont=dict(color='white', size=10),
                customdata=[[lu['player_name'], lu['BB%'], lu['Barrel Rate'],
                             lu['In Zone Swing %'], lu['Out of Zone Swing %'], lu['Whiff %']]],
                hovertemplate=(
                    '<b>%{customdata[0]}</b><br>'
                    'BB%%: %{customdata[1]:.1f}  |  Barrel: %{customdata[2]:.1f}<br>'
                    'IZ Swing: %{customdata[3]:.1f}  |  OZ Swing: %{customdata[4]:.1f}<br>'
                    'Whiff: %{customdata[5]:.1f}'
                    '<extra></extra>'
                ),
                showlegend=False,
            ))

    if new_player_pca is not None:
        fig.add_trace(go.Scatter(
            x=[new_player_pca[0, 0]],
            y=[new_player_pca[0, 1]],
            mode='markers+text',
            name=player_name,
            marker=dict(color='#EF4444', size=18, symbol='star'),
            text=[player_name],
            textposition='top right',
            textfont=dict(color='#EF4444', size=11),
            hovertemplate=(
                f'<b>{player_name}</b><br>'
                f'BB%%: {bb:.1f}  |  Barrel: {barrel:.1f}<br>'
                f'IZ Swing: {iz_swing:.1f}  |  OZ Swing: {oz_swing:.1f}<br>'
                f'Whiff: {whiff:.1f}'
                '<extra></extra>'
            ),
        ))

    fig.update_layout(
        paper_bgcolor='#111827',
        plot_bgcolor='#111827',
        font=dict(color='#9CA3AF'),
        xaxis=dict(
            title='← Better Plate Discipline    |    More Aggressive →',
            gridcolor='#374151', zerolinecolor='#374151',
            color='#9CA3AF', title_font=dict(size=11),
        ),
        yaxis=dict(
            title='← Contact Focus    |    Higher Power →',
            gridcolor='#374151', zerolinecolor='#374151',
            color='#9CA3AF', title_font=dict(size=11),
        ),
        legend=dict(
            bgcolor='#1F2937', bordercolor='#374151', borderwidth=1,
            font=dict(color='white', size=10),
        ),
        hoverlabel=dict(bgcolor='#1F2937', bordercolor='#374151', font=dict(color='white')),
        height=480,
        margin=dict(l=10, r=10, t=20, b=10),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Result card
    if new_archetype:
        color = ARCHETYPE_COLORS[new_archetype]
        desc  = ARCHETYPE_DESCRIPTIONS[new_archetype]
        similar_html = '  •  '.join(
            f'<span style="color:white; font-weight:600;">{p}</span>'
            for p in similar_players
        )

        stat_inputs = {
            'BB%': bb, 'Barrel Rate': barrel,
            'In Zone Swing %': iz_swing, 'Out of Zone Swing %': oz_swing,
            'Whiff %': whiff,
        }
        bars_html = ''
        for feat, val in stat_inputs.items():
            pct = (df_model[feat] <= val).sum() / len(df_model) * 100
            bars_html += f'''
            <div style="display:flex;align-items:center;margin-bottom:5px;gap:8px;">
              <div style="width:115px;color:#D1D5DB;font-size:12px;white-space:nowrap;">{feat}</div>
              <div style="flex:1;background:#374151;border-radius:3px;height:7px;overflow:hidden;">
                <div style="width:{pct:.0f}%;background:{color};height:100%;border-radius:3px;"></div>
              </div>
              <div style="width:105px;color:#9CA3AF;font-size:11px;text-align:right;">{val:.1f} · {_ordinal(pct)} pctile</div>
            </div>'''

        st.markdown(f"""
        <div style="background:#1F2937; border-left:4px solid {color};
                    padding:16px 20px; border-radius:6px; margin-top:8px;">
            <div style="color:#9CA3AF; font-size:12px; margin-bottom:4px;">CLASSIFICATION RESULT</div>
            <div style="color:{color}; font-size:22px; font-weight:700;">{new_archetype}</div>
            <div style="color:#D1D5DB; font-size:13px; margin-top:4px;">{desc}</div>
            <div style="color:#9CA3AF; font-size:12px; margin-top:10px;">
                Most similar players: {similar_html}
            </div>
            <div style="margin-top:14px; border-top:1px solid #374151; padding-top:12px;">
              <div style="color:#9CA3AF; font-size:11px; margin-bottom:8px;
                          text-transform:uppercase; letter-spacing:0.05em;">
                Stat Percentiles vs. {len(df_model)} qualified players
              </div>
              {bars_html}
            </div>
        </div>
        """, unsafe_allow_html=True)

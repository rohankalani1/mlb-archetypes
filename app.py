import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# ── Header ────────────────────────────────────────────────────────────────────
st.title("⚾ MLB Batter Archetype Classifier")
st.caption("K-Means clustering on 2021–2025 Statcast data  •  106 qualified players  •  4 archetypes")
st.divider()

# ── Layout ────────────────────────────────────────────────────────────────────
left, right = st.columns([1, 2.2])

with left:
    st.subheader("Player Input")

    player_name = st.text_input("Player Name", placeholder="e.g. Shohei Ohtani")

    st.markdown("**Batting Stats**")
    bb       = st.number_input("BB%",         min_value=0.0,  max_value=30.0, value=8.5,  step=0.1, help="Walk rate")
    barrel   = st.number_input("Barrel Rate", min_value=0.0,  max_value=30.0, value=10.2, step=0.1, help="Barrel batted ball rate")
    iz_swing = st.number_input("IZ Swing %",  min_value=40.0, max_value=90.0, value=68.4, step=0.1, help="In-zone swing rate")
    oz_swing = st.number_input("OZ Swing %",  min_value=5.0,  max_value=55.0, value=28.1, step=0.1, help="Out-of-zone chase rate")
    whiff    = st.number_input("Whiff %",     min_value=5.0,  max_value=50.0, value=24.3, step=0.1, help="Miss rate on all swings")

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

    new_player_pca = None
    new_archetype  = None
    nearest_player = None

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

        cluster_mask   = df_model['Cluster'].values == cluster_id
        dists          = cdist(new_sc, X_sc[cluster_mask], metric='euclidean')[0]
        nearest_player = df_model[cluster_mask].iloc[dists.argmin()]['player_name']

    elif classify and not player_name.strip():
        st.warning("Enter a player name before classifying.")

    # Draw map
    fig, ax = plt.subplots(figsize=(9, 5.8), facecolor='#111827')
    ax.set_facecolor('#111827')

    for arch, color in ARCHETYPE_COLORS.items():
        m = df_model['Archetype'] == arch
        ax.scatter(df_model[m]['PCA1'], df_model[m]['PCA2'],
                   color=color, s=65, alpha=0.72, label=arch, edgecolors='none')

    notable = ['Soto, Juan', 'Judge, Aaron', 'Perez, Salvador',
               'Ohtani, Shohei', 'Arraez, Luis', 'Kwan, Steven', 'Báez, Javier']
    for _, row in df_model[df_model['player_name'].isin(notable)].iterrows():
        ax.annotate(
            row['player_name'].split(',')[0],
            (row['PCA1'], row['PCA2']),
            textcoords='offset points', xytext=(6, 4),
            fontsize=8, color='white', alpha=0.8
        )

    if new_player_pca is not None:
        star_color = ARCHETYPE_COLORS[new_archetype]
        ax.scatter(new_player_pca[0, 0], new_player_pca[0, 1],
                   color='#EF4444', s=280, zorder=6, marker='*', label=player_name)
        ax.annotate(
            player_name,
            (new_player_pca[0, 0], new_player_pca[0, 1]),
            textcoords='offset points', xytext=(9, 6),
            fontsize=9, color='#EF4444', fontweight='bold'
        )

    ax.set_xlabel('← Better Plate Discipline    |    More Aggressive →',
                  color='#9CA3AF', fontsize=10)
    ax.set_ylabel('← Contact Focus    |    Higher Power →',
                  color='#9CA3AF', fontsize=10)
    ax.legend(loc='upper left', framealpha=0.2, labelcolor='white',
              facecolor='#1F2937', edgecolor='#374151', fontsize=9)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    ax.spines['left'].set_color('#374151')
    ax.spines['bottom'].set_color('#374151')
    ax.tick_params(colors='#9CA3AF')
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    # Result card
    if new_archetype:
        color = ARCHETYPE_COLORS[new_archetype]
        desc  = ARCHETYPE_DESCRIPTIONS[new_archetype]
        st.markdown(f"""
        <div style="background:#1F2937; border-left:4px solid {color};
                    padding:16px 20px; border-radius:6px; margin-top:8px;">
            <div style="color:#9CA3AF; font-size:12px; margin-bottom:4px;">CLASSIFICATION RESULT</div>
            <div style="color:{color}; font-size:22px; font-weight:700;">{new_archetype}</div>
            <div style="color:#D1D5DB; font-size:13px; margin-top:4px;">{desc}</div>
            <div style="color:#9CA3AF; font-size:12px; margin-top:10px;">
                Most similar to: <span style="color:white; font-weight:600;">{nearest_player}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

# --------- Chargement et Préparation ---------
@st.cache_data
def load_data():
    df = pd.read_csv("data/owa_action_fact.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='s')
    df["type_contenu"] = df["action_label"].fillna("").apply(
        lambda x: "article" if "article" in x.lower()
        else "dataset" if "dataset" in x.lower()
        else "user" if "user" in x.lower()
        else "autre"
    )
    df["is_engaging"] = df["action_name"].isin(["create", "publish", "modify"])
    df["is_social"] = df["type_contenu"] == "user"
    return df

def calculate_kpis(df):
    grouped = df.groupby("visitor_id").agg(
        nb_total_actions=("id", "count"),
        nb_sessions_uniques=("session_id", pd.Series.nunique),
        nb_types_actions=("action_name", pd.Series.nunique),
        nb_actions_engageantes=("is_engaging", "sum"),
        nb_articles=("type_contenu", lambda x: (x == "article").sum()),
        nb_datasets=("type_contenu", lambda x: (x == "dataset").sum()),
        nb_user_target=("type_contenu", lambda x: (x == "user").sum()),
        date_dernière_action=("timestamp", "max"),
        date_première_action=("timestamp", "min")
    ).reset_index()

    now = pd.Timestamp(datetime.now())
    grouped["jours_depuis_dernière_action"] = (now - grouped["date_dernière_action"]).dt.days
    grouped["ratio_engaging"] = grouped["nb_actions_engageantes"] / grouped["nb_total_actions"]
    grouped["diversité_contenu"] = grouped[["nb_articles", "nb_datasets", "nb_user_target"]].std(axis=1)
    grouped["actions_par_session"] = grouped["nb_total_actions"] / grouped["nb_sessions_uniques"]
    return grouped

def compute_sec_ics(kpi_df):
    df = kpi_df.copy()
    df["recence_inverse"] = df["jours_depuis_dernière_action"].max() - df["jours_depuis_dernière_action"]
    scaler = MinMaxScaler()
    cols = ["nb_total_actions", "nb_sessions_uniques", "nb_types_actions", "ratio_engaging", "recence_inverse"]
    df_norm = pd.DataFrame(scaler.fit_transform(df[cols]), columns=cols)
    df["SEC"] = df_norm.mean(axis=1) * 100
    df["ICS"] = df["nb_actions_engageantes"] / df["nb_total_actions"]
    return df

def compute_ira_srd(df, kpi_df):
    kpi = kpi_df.copy()
    actions = df[["visitor_id", "timestamp"]].sort_values(["visitor_id", "timestamp"])
    def compute_intervals(group):
        deltas = group["timestamp"].diff().dt.days.dropna()
        return pd.Series({
            "mean_days_between_actions": deltas.mean() if not deltas.empty else np.nan,
            "std_days_between_actions": deltas.std() if not deltas.empty else np.nan
        })

    intervals = actions.groupby("visitor_id").apply(compute_intervals).reset_index()
    kpi = kpi.merge(intervals, on="visitor_id", how="left")
    kpi["mean_days_inv"] = kpi["mean_days_between_actions"].max() - kpi["mean_days_between_actions"]
    ira_data = kpi[["mean_days_inv", "std_days_between_actions"]].fillna(0)
    kpi["IRA"] = MinMaxScaler().fit_transform(ira_data).mean(axis=1) * 100

    srd_data = pd.DataFrame({
        "SEC_inv": 100 - kpi["SEC"],
        "IRA_inv": 100 - kpi["IRA"],
        "recence": kpi["jours_depuis_dernière_action"]
    }).fillna(0)
    kpi["SRD"] = MinMaxScaler().fit_transform(srd_data).mean(axis=1) * 100
    return kpi

# --------- UI Setup ---------
st.set_page_config(page_title="Moteur de Recommandation", layout="wide")
page = st.sidebar.radio("📂 Navigation", ["🧭 Indicateurs clés", "📊 Analyse comportementale", "🚨 Recommandations stratégiques"])

df = load_data()
kpi = calculate_kpis(df)
kpi = compute_sec_ics(kpi)
kpi = compute_ira_srd(df, kpi)

# --------- PAGE 1 ---------
if page == "🧭 Indicateurs clés":
    st.title("🧭 Diagnostic : Indicateurs d'engagement")
    st.markdown("""
    Cette page présente les **indicateurs clés** calculés pour chaque utilisateur afin de piloter l'engagement de la plateforme :
    - **SEC** : Score global d'engagement sur 100 (volume, diversité, récence, actions actives)
    - **ICS** : % d’actions structurantes (create, publish, modify)
    - **IRA** : Stabilité et fréquence des visites
    - **SRD** : Risque calculé de désengagement
    """)

    col1, col2, col3, col4 = st.columns(4)
    for title, val in zip(["SEC", "ICS", "IRA", "SRD"],
                          [kpi["SEC"].mean(), kpi["ICS"].mean()*100, kpi["IRA"].mean(), kpi["SRD"].mean()]):
        col = locals()[f'col{["SEC", "ICS", "IRA", "SRD"].index(title)+1}']
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=val,
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "royalblue"}},
            title={"text": title}
        ))
        col.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("📌 Histogramme des scores SEC")
    st.plotly_chart(px.histogram(kpi, x="SEC", nbins=20, color_discrete_sequence=["royalblue"]))

# --------- PAGE 2 ---------
elif page == "📊 Analyse comportementale":
    st.title("📊 Analyse approfondie des comportements")
    st.markdown("Analyse croisée des indicateurs pour détecter des profils types ou comportements divergents.")
    
    st.plotly_chart(px.scatter(kpi, x="SEC", y="IRA", color="ICS",
                               title="Engagement vs Régularité", hover_name="visitor_id"))

    st.plotly_chart(px.histogram(kpi, x="diversité_contenu", nbins=15,
                                 title="Spécialisation ou diversité des contenus"))

    df_merge = df.merge(kpi, on="visitor_id")
    group_scores = df_merge.groupby("action_group")[["SEC", "ICS", "IRA", "SRD"]].mean().reset_index()
    st.plotly_chart(px.bar(group_scores, x="action_group", y=["SEC", "IRA", "SRD"],
                           barmode="group", title="Moyenne des scores par action_group"))

# --------- PAGE 3 ---------
elif page == "🚨 Recommandations stratégiques":
    st.title("🚨 Recommandations et alertes utilisateurs")
    st.markdown("""
    Utilisez cette page pour identifier :
    - 🔴 **Utilisateurs à risque** (SRD élevé)
    - 🟢 **Utilisateurs à fort engagement** (SEC élevé)
    - 🟡 **Utilisateurs stables mais à surveiller**
    """)

    top_risk = kpi[kpi["SRD"] > 75].sort_values("SRD", ascending=False).head(10)
    top_active = kpi[kpi["SEC"] > 80].sort_values("SEC", ascending=False).head(10)

    col1, col2 = st.columns(2)
    col1.metric("Utilisateurs à risque", len(top_risk))
    col2.metric("Utilisateurs très engagés", len(top_active))

    st.subheader("🔴 À relancer")
    st.dataframe(top_risk[["visitor_id", "SEC", "IRA", "SRD"]])

    st.subheader("🟢 À valoriser")
    st.dataframe(top_active[["visitor_id", "SEC", "ICS", "IRA"]])

    st.markdown("---")
    st.markdown("📌 **Actions recommandées :**")
    st.markdown("""
    - Relancer les utilisateurs avec **SRD > 75**
    - Récompenser ceux avec **SEC > 80** et **ICS > 0.5**
    - Surveiller ceux avec **IRA > 70** mais **SEC < 50**
    """)
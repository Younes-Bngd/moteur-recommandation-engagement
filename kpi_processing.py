import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def load_and_process_data(df):
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='s')
    df["type_contenu"] = df["action_label"].fillna("").apply(
        lambda x: "article" if "article" in x.lower()
        else "dataset" if "dataset" in x.lower()
        else "user" if "user" in x.lower()
        else "autre"
    )
    engaging_actions = ["create", "publish", "modify"]
    df["is_engaging"] = df["action_name"].isin(engaging_actions)
    df["is_social"] = df["type_contenu"] == "user"
    return df

def calculate_user_kpis(df):
    grouped = df.groupby("visitor_id").agg(
        nb_total_actions=("id", "count"),
        nb_sessions_uniques=("session_id", pd.Series.nunique),
        nb_types_actions=("action_name", pd.Series.nunique),
        nb_actions_engageantes=("is_engaging", "sum"),
        nb_actions_sociales=("is_social", "sum"),
        nb_articles=("type_contenu", lambda x: (x == "article").sum()),
        nb_datasets=("type_contenu", lambda x: (x == "dataset").sum()),
        nb_user_target=("type_contenu", lambda x: (x == "user").sum()),
        date_dernière_action=("timestamp", "max"),
        date_première_action=("timestamp", "min")
    ).reset_index()

    now = pd.Timestamp(datetime.now())
    grouped["jours_depuis_dernière_action"] = (now - grouped["date_dernière_action"]).dt.days
    grouped["ancienneté_utilisateur"] = (now - grouped["date_première_action"]).dt.days
    grouped["ratio_engaging"] = grouped["nb_actions_engageantes"] / grouped["nb_total_actions"]
    grouped["ratio_social"] = grouped["nb_actions_sociales"] / grouped["nb_total_actions"]
    grouped["diversité_contenu"] = grouped[["nb_articles", "nb_datasets", "nb_user_target"]].std(axis=1)
    grouped["actions_par_session"] = grouped["nb_total_actions"] / grouped["nb_sessions_uniques"]
    return grouped

def compute_sec_ics(kpi_df):
    df = kpi_df.copy()
    df["recence_inverse"] = df["jours_depuis_dernière_action"].max() - df["jours_depuis_dernière_action"]
    cols = ["nb_total_actions", "nb_sessions_uniques", "nb_types_actions", "ratio_engaging", "recence_inverse"]
    scaler = MinMaxScaler()
    df_norm = pd.DataFrame(scaler.fit_transform(df[cols]), columns=cols)
    df["SEC"] = df_norm.mean(axis=1) * 100
    df["ICS"] = df["nb_actions_engageantes"] / df["nb_total_actions"]
    return df

def compute_ira_srd(df_actions, kpi_df):
    df_kpi = kpi_df.copy()
    actions = df_actions[["visitor_id", "timestamp"]].sort_values(["visitor_id", "timestamp"])

    def compute_intervals(group):
        deltas = group["timestamp"].diff().dt.days.dropna()
        return pd.Series({
            "mean_days_between_actions": deltas.mean() if not deltas.empty else np.nan,
            "std_days_between_actions": deltas.std() if not deltas.empty else np.nan
        })

    intervals = actions.groupby("visitor_id").apply(compute_intervals).reset_index()
    df_kpi = df_kpi.merge(intervals, on="visitor_id", how="left")

    df_kpi["mean_days_inv"] = df_kpi["mean_days_between_actions"].max() - df_kpi["mean_days_between_actions"]
    ira_comp = df_kpi[["mean_days_inv", "std_days_between_actions"]].copy()
    ira_comp["std_days_between_actions"] = df_kpi["std_days_between_actions"].max() - df_kpi["std_days_between_actions"]
    ira_scaled = MinMaxScaler().fit_transform(ira_comp.fillna(0))
    df_kpi["IRA"] = ira_scaled.mean(axis=1) * 100

    srd_comp = pd.DataFrame()
    srd_comp["jours_depuis_dernière_action"] = df_kpi["jours_depuis_dernière_action"]
    srd_comp["IRA_inv"] = 100 - df_kpi["IRA"]
    srd_comp["SEC_inv"] = 100 - df_kpi["SEC"]
    srd_scaled = MinMaxScaler().fit_transform(srd_comp.fillna(0))
    df_kpi["SRD"] = srd_scaled.mean(axis=1) * 100

    return df_kpi
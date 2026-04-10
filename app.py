import io
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="Plateforme de Clustering K-Means",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
        .main {background-color: #f8fafc;}
        .block-container {padding-top: 1.5rem;}
        .metric-card {
            background: white;
            padding: 1rem;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_csv(uploaded_file, separator: Optional[str] = None) -> pd.DataFrame:
    """Charge un CSV en essayant plusieurs séparateurs si nécessaire."""
    content = uploaded_file.getvalue()
    candidates = [separator] if separator else [None, ",", ";", "\t", "|"]

    for sep in candidates:
        try:
            if sep is None:
                df = pd.read_csv(io.BytesIO(content), sep=None, engine="python")
            else:
                df = pd.read_csv(io.BytesIO(content), sep=sep)
            if not df.empty:
                return df
        except Exception:
            continue

    return pd.read_csv(io.BytesIO(content))


def make_frequency_plot(df: pd.DataFrame, column: str):
    series = df[column].dropna()
    if series.empty:
        st.warning("Aucune donnée exploitable pour cette colonne.")
        return

    if pd.api.types.is_numeric_dtype(series):
        bins = st.slider("Nombre d'intervalles", 5, 30, 10)
        fig = px.histogram(
            df,
            x=column,
            nbins=bins,
            title=f"Distribution de {column}",
            color_discrete_sequence=["#2563eb"],
        )
    else:
        counts = series.astype(str).value_counts().reset_index()
        counts.columns = [column, "Fréquence"]
        fig = px.bar(
            counts,
            x=column,
            y="Fréquence",
            title=f"Fréquence par {column}",
            color="Fréquence",
            color_continuous_scale="Blues",
        )

    st.plotly_chart(fig, use_container_width=True)


def auto_convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convertit automatiquement les colonnes textuelles numériques pour faciliter l'analyse."""
    converted_df = df.copy()
    for column in converted_df.select_dtypes(include="object").columns:
        non_null = converted_df[column].dropna()
        if non_null.empty:
            continue

        cleaned = (
            non_null.astype(str)
            .str.replace("\u202f", "", regex=False)
            .str.replace(" ", "", regex=False)
            .str.replace(",", ".", regex=False)
        )
        numeric_series = pd.to_numeric(cleaned, errors="coerce")

        if numeric_series.notna().sum() >= max(3, int(0.6 * len(non_null))):
            converted_df.loc[non_null.index, column] = numeric_series
            converted_df[column] = pd.to_numeric(converted_df[column], errors="coerce")

    return converted_df


def format_column_preview(columns: list[str], limit: int = 8) -> str:
    if not columns:
        return "Aucune"

    preview = columns[:limit]
    suffix = " ..." if len(columns) > limit else ""
    return ", ".join(f"`{col}`" for col in preview) + suffix


st.title("📊 Plateforme interactive de clustering K-Means")
st.caption(
    "Importez plusieurs fichiers CSV, explorez vos données et créez des clusters de manière fluide."
)

with st.expander("ℹ️ Comment utiliser la plateforme", expanded=True):
    st.markdown(
        """
        1. **Importez** un ou plusieurs fichiers CSV.
        2. Si les colonnes diffèrent selon les fichiers, choisissez **Un fichier précis**.
        3. Sélectionnez une colonne pour voir sa **fréquence** (`age`, `ville`, etc.).
        4. Choisissez au moins **2 colonnes numériques** pour lancer le **K-Means**.
        """
    )

with st.sidebar:
    st.header("⚙️ Paramètres")
    separator = st.selectbox(
        "Séparateur CSV",
        options=["Auto", ",", ";", "Tab", "|"],
        index=0,
    )
    separator_map = {"Auto": None, ",": ",", ";": ";", "Tab": "\t", "|": "|"}

    uploaded_files = st.file_uploader(
        "Importer un ou plusieurs fichiers CSV",
        type=["csv"],
        accept_multiple_files=True,
    )

if not uploaded_files:
    st.info(
        "👈 Commencez par importer un ou plusieurs fichiers CSV depuis la barre latérale pour lancer l'analyse."
    )
    st.stop()

frames = []
load_errors = []
for uploaded_file in uploaded_files:
    try:
        df = load_csv(uploaded_file, separator_map[separator]).copy()
        df["source_file"] = uploaded_file.name
        frames.append(df)
    except Exception as exc:
        load_errors.append(f"{uploaded_file.name}: {exc}")

if load_errors:
    for err in load_errors:
        st.error(err)

if not frames:
    st.warning("Aucun fichier CSV valide n'a pu être chargé.")
    st.stop()

combined_df = pd.concat(frames, ignore_index=True, sort=False)
combined_df = auto_convert_numeric_columns(combined_df)

st.subheader("🧭 Étape 1 : choisissez les données à analyser")
source_options = sorted(combined_df["source_file"].dropna().astype(str).unique().tolist())

if len(source_options) > 1:
    analysis_mode = st.radio(
        "Mode d'analyse",
        options=["Tous les fichiers", "Un fichier précis"],
        horizontal=True,
        help="Si vos CSV sont différents, choisissez 'Un fichier précis' pour voir les bonnes colonnes plus facilement.",
    )
else:
    analysis_mode = "Un fichier précis"

if analysis_mode == "Un fichier précis":
    selected_source = st.selectbox(
        "Fichier à utiliser",
        source_options,
        help="Choisissez ici le CSV sur lequel vous voulez faire l'analyse et le clustering.",
    )
    active_df = combined_df[combined_df["source_file"].astype(str) == selected_source].copy()
else:
    selected_source = "Tous les fichiers"
    active_df = combined_df.copy()

st.subheader("🔎 Aperçu général")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Fichiers chargés", len(frames))
col2.metric("Lignes visibles", f"{active_df.shape[0]:,}".replace(",", " "))
col3.metric("Colonnes visibles", active_df.shape[1])
col4.metric("Colonnes numériques", len(active_df.select_dtypes(include=np.number).columns))

numeric_preview = [col for col in active_df.select_dtypes(include=np.number).columns.tolist() if col != "source_file"]
other_preview = [col for col in active_df.columns.tolist() if col not in numeric_preview]

st.info(
    f"**Sélection actuelle :** {selected_source}  \n"
    f"**Colonnes numériques détectées :** {format_column_preview(numeric_preview)}  \n"
    f"**Autres colonnes :** {format_column_preview(other_preview)}"
)

with st.expander("Voir les données affichées", expanded=True):
    st.dataframe(active_df.head(200), use_container_width=True)

filtered_df = active_df.copy()
filterable_columns = [
    c for c in filtered_df.columns
    if filtered_df[c].nunique(dropna=True) <= 50 and c != "source_file"
]
if filterable_columns:
    st.subheader("🎛️ Étape 2 : filtres interactifs")
    selected_filter = st.selectbox(
        "Filtrer sur une colonne",
        ["Aucun"] + filterable_columns,
        help="Optionnel : gardez uniquement certaines valeurs avant l'analyse.",
    )
    if selected_filter != "Aucun":
        values = sorted(filtered_df[selected_filter].dropna().astype(str).unique().tolist())
        chosen_values = st.multiselect(
            f"Valeurs à garder pour {selected_filter}",
            values,
            default=values,
        )
        if chosen_values:
            filtered_df = filtered_df[filtered_df[selected_filter].astype(str).isin(chosen_values)]

st.subheader("📈 Étape 3 : analyse de fréquence")
frequency_options = [col for col in filtered_df.columns.tolist() if col != "source_file"]
if not frequency_options:
    frequency_options = filtered_df.columns.tolist()

default_frequency_index = frequency_options.index("age") if "age" in frequency_options else 0
frequency_column = st.selectbox(
    "Choisissez une colonne à analyser (ex: age)",
    frequency_options,
    index=default_frequency_index,
    help="Exemples : `age`, `ville`, `genre`, `profession`, etc.",
)
make_frequency_plot(filtered_df, frequency_column)

st.subheader("🤖 Étape 4 : clustering K-Means")
numeric_columns = [col for col in filtered_df.select_dtypes(include=np.number).columns.tolist() if col != "source_file"]

if len(numeric_columns) < 2:
    st.warning("Je n'ai pas trouvé au moins deux colonnes numériques dans la sélection actuelle.")
    st.markdown(
        """
        **Pour débloquer le clustering :**
        - choisissez **Un fichier précis** dans l'étape 1 ;
        - prenez un CSV contenant des colonnes comme `age`, `revenu`, `score` ;
        - sélectionnez ensuite au moins **2 variables numériques**.
        """
    )
    st.stop()

recommended_features = (
    filtered_df[numeric_columns]
    .notna()
    .sum()
    .sort_values(ascending=False)
    .head(min(4, len(numeric_columns)))
    .index.tolist()
)

st.caption(
    "Colonnes recommandées : "
    + ", ".join(f"`{col}`" for col in recommended_features)
)

selected_features = st.multiselect(
    "Sélectionnez les variables numériques pour le clustering",
    numeric_columns,
    default=recommended_features[: min(3, len(recommended_features))],
    help="Choisissez au moins 2 colonnes numériques. La recherche dans la liste est possible.",
)

if len(selected_features) < 2:
    st.info("Sélectionnez au moins deux variables pour créer des clusters.")
    st.stop()

n_clusters = st.slider("Nombre de clusters (K)", 2, min(10, max(2, len(filtered_df) - 1)), 3)
normalize = st.checkbox("Normaliser les variables avant clustering", value=True)

work_df = filtered_df[selected_features].copy()
work_df = work_df.replace([np.inf, -np.inf], np.nan)
work_df = work_df.dropna()

if work_df.shape[0] <= n_clusters:
    st.warning("Pas assez de lignes exploitables après nettoyage pour le nombre de clusters choisi.")
    st.stop()

X = work_df.values
if normalize:
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels = model.fit_predict(X)

clustered_df = filtered_df.loc[work_df.index].copy()
clustered_df["cluster"] = labels.astype(str)

score = None
if len(set(labels)) > 1 and len(clustered_df) > n_clusters:
    score = silhouette_score(X, labels)

metric_a, metric_b = st.columns(2)
metric_a.metric("Observations clusterisées", len(clustered_df))
metric_b.metric("Score silhouette", f"{score:.3f}" if score is not None else "N/A")

summary = clustered_df.groupby("cluster")[selected_features].mean(numeric_only=True)
st.write("### Moyenne des variables par cluster")
st.dataframe(summary.style.format("{:.2f}"), use_container_width=True)

plot_df = pd.DataFrame(X, columns=selected_features)
if len(selected_features) > 2:
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(X)
    viz_df = pd.DataFrame(components, columns=["Dimension 1", "Dimension 2"])
else:
    viz_df = plot_df.iloc[:, :2].copy()
    viz_df.columns = ["Dimension 1", "Dimension 2"]

viz_df["cluster"] = labels.astype(str)
scatter = px.scatter(
    viz_df,
    x="Dimension 1",
    y="Dimension 2",
    color="cluster",
    title="Visualisation des clusters",
    template="plotly_white",
)
st.plotly_chart(scatter, use_container_width=True)

csv_data = clustered_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Télécharger les données clusterisées",
    data=csv_data,
    file_name="resultats_clustering.csv",
    mime="text/csv",
)

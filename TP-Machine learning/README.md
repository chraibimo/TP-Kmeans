# Plateforme de clustering K-Means

Application interactive en **Streamlit** pour :
- importer **plusieurs fichiers CSV** à la fois,
- explorer les données,
- analyser la **fréquence** d'une colonne choisie (`age`, `ville`, etc.),
- lancer un **clustering K-Means** sur les variables numériques sélectionnées,
- visualiser et télécharger les résultats.

## Lancer le projet

```powershell
pip install -r requirements.txt
streamlit run app.py
```

## Fonctionnalités

- Upload multiple CSV
- Fusion automatique avec colonne `source_file`
- Prévisualisation des données
- Analyse de fréquence interactive
- Clustering K-Means avec choix des variables
- Visualisation 2D des clusters
- Téléchargement du dataset enrichi

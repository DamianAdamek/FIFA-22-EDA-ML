import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from utils import reader
from EDA import stats

plots_prefix = Path('plots')
plots_prefix.mkdir(parents=True, exist_ok=True)

base_df = reader.read_data('../utils/players_22')
stats.count_stats(base_df)
plot_size = (10,10)

top_10_leagues = base_df[base_df['league_name'].isin(['French Ligue 1', 
                                         'German 1. Bundesliga',
                                         'English Premier League',
                                         'Spain Primera Division',
                                         'Italian Serie A',
                                         'Holland Eredivisie',
                                         'Portuguese Liga ZON SAGRES',
                                         'Belgian Jupiler Pro League',
                                         'Czech Republic Gambrinus Liga',
                                         'USA Major League Soccer'])]

custom_10_leagues = base_df[base_df['league_name'].isin(['Holland Eredivisie',
                                         'Danish Superliga',
                                         'Austrian Football Bundesliga',
                                         'Rep. Ireland Airtricity League',
                                         'Belgian Jupiler Pro League',
                                         'Turkish Süper Lig',
                                         'Korean K League 1',
                                         'Chinese Super League',
                                         'Paraguayan Primera División',
                                         'South African Premier Division'])]

nations = base_df[base_df['nationality_name'].isin(['Poland',
                                            'Japan',
                                            'France',
                                            'Spain',
                                            'Argentina',
                                            'Brazil',
                                            'England',
                                            'Korea Republic',
                                            'Netherlands',
                                            'Italy'])]

positions_map = {
    'LW': 'Skrzydłowy',
    'RW': 'Skrzydłowy',
    'LM': 'Skrzydłowy',
    'RM': 'Skrzydłowy',
    'ST': 'Napastnik',
    'CF': 'Napastnik',
    'CAM': 'Pomocnik',
    'CM': 'Pomocnik',
    'CDM': 'Pomocnik',
    'CB': 'Obrońca',
    'LB': 'Boczny obrońca',
    'RB': 'Boczny obrońca',
    'LWB': 'Boczny obrońca',
    'RWB': 'Boczny obrońca',
    'GK': 'Bramkarz'
}


df_pos = base_df[['player_positions', 'pace', 'shooting', 'dribbling', 'passing','defending', 'physic']].dropna()
df_pos = df_pos.explode('player_positions')
df_pos['position_group'] = df_pos['player_positions'].map(positions_map)
traits = ['pace', 'shooting', 'dribbling', 'passing', 'defending', 'physic']
df_box = df_pos[['position_group'] + traits].dropna()
df_melt = df_box.melt(id_vars='position_group', var_name='trait', value_name='value')
plt.figure(figsize=plot_size)
sns.boxplot(data=df_melt,
            x='trait',
            y='value',
            hue='position_group')
plt.title('Zależność cech piłkarza od pozycji na jakiej gra')
plt.xlabel('Cecha')
plt.ylabel('Wartość')
plt.legend(title='Grupa pozycji', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(plots_prefix / 'boxplot_traits_vs_pozycja.png', dpi=300)


plt.figure(figsize=plot_size)
sns.violinplot(data=nations, 
                y='nationality_name',
                x='age')
plt.title('Zależność wieku piłkarza od kraju pochodzenia')
plt.xlabel('Wiek')
plt.ylabel('Narodowość')
plt.tight_layout()
plt.savefig(plots_prefix / 'violinplot_nardowosc_vs_wiek.png', dpi=300)


plt.figure(figsize=plot_size)
sns.histplot(top_10_leagues,
            x='league_name')
plt.xticks(rotation=45)
plt.title('Liczba piłkarzy w lidze')
plt.xlabel('Ligi')
plt.ylabel('Liczba piłkarzy w lidze')
plt.tight_layout()
plt.savefig(plots_prefix / 'histogram_ligi.png', dpi=300)


plt.figure(figsize=plot_size)
df_his = base_df[['player_positions', 'preferred_foot']].dropna()
df_his = df_his.explode('player_positions')
sns.histplot(df_his,
            x='player_positions',
            hue='preferred_foot')
plt.title('Histogram pozycji piłkarzy')
plt.xlabel('Pozycje')
plt.ylabel('Liczba piłkarzy na tej pozycji')
plt.tight_layout()
plt.savefig(plots_prefix / 'histogram_pozycji.png', dpi=300)


plt.figure(figsize=plot_size)
sns.barplot(custom_10_leagues,
            x='league_name',
            y='age',
            errorbar='sd')
plt.xticks(rotation=45)
plt.title('Zależność między wiekiem piłkarza a ligą w jakiej gra')
plt.xlabel('Ligi')
plt.ylabel('Wiek')
plt.tight_layout()
plt.savefig(plots_prefix / 'barplot_liga_vs_wiek.png', dpi=300)



plt.figure(figsize=plot_size)
df_heatmap = nations[['nationality_name', 'player_positions']].dropna()
df_heatmap = df_heatmap.explode('player_positions')
df_heatmap['position_group'] = df_heatmap['player_positions'].map(positions_map)
positions_by_country = df_heatmap.groupby(['nationality_name', 'position_group']).size().reset_index(name='count')
pivot = positions_by_country.pivot(index='position_group', columns='nationality_name', values='count').fillna(0)
position_counts = df_heatmap['position_group'].value_counts()
sorted_positions = position_counts.index.tolist()
pivot = pivot.loc[sorted_positions]
nationalities_counts = df_heatmap['nationality_name'].value_counts()
sorted_nationalities = nationalities_counts.index.tolist()
pivot = pivot[sorted_nationalities]
sns.heatmap(pivot,
            cmap='viridis',
            linewidths=0.5,
            annot=True,
            fmt='.0f')
plt.xticks(rotation=45)
plt.title('Pozycje piłkarzy w danym kraju')
plt.tight_layout()
plt.savefig(plots_prefix / 'heatmap_kraj_vs_pozycja.png', dpi=300)


plt.figure(figsize=plot_size)
sns.regplot(data=base_df,
            x='height_cm',
            y='weight_kg',
            ci=95)
plt.title('Wzrost piłkarza a waga')
plt.xlabel('Wzrost')
plt.ylabel('Waga')
plt.tight_layout()
plt.savefig(plots_prefix / 'regplot_wiek_vs_waga.png', dpi=300)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df_pca_input = base_df[['player_positions'] + traits].dropna()
df_pca_input = df_pca_input.explode('player_positions')
df_pca_input['position_group'] = df_pca_input['player_positions'].map(positions_map)

df_pca_input = df_pca_input.dropna(subset=['position_group'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_pca_input[traits])

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df_pca_result = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca_result['position_group'] = df_pca_input['position_group'].values

plt.figure(figsize=(10, 8))
sns.scatterplot(data=df_pca_result, x='PC1', y='PC2', hue='position_group', alpha=0.7)
plt.title('Redukcja cech piłkarzy za pomocą PCA')
plt.tight_layout()
plt.savefig(plots_prefix / 'pca_cechy_grupy_pozycji.png', dpi=300)

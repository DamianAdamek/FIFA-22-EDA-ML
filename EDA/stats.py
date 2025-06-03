import pandas as pd
from pathlib import Path

def count_stats(data: pd.DataFrame):
    # for numerical features
    labels_num = []
    means = []
    medians = []
    min_vals = []
    max_vals = []
    std_devs = []
    percentiles_5 = []
    percentiles_95 = []
    missing_counts_num = []

    # for categorical features
    labels_cat = []
    unique_counts = []
    missing_counts_cat = []
    proportions = []

    for col_name in data.columns:
        col = data[col_name]

        if pd.api.types.is_numeric_dtype(col):
            labels_num.append(col_name)
            means.append(col.mean())
            medians.append(col.median())
            min_vals.append(col.min())
            max_vals.append(col.max())
            std_devs.append(col.std())
            percentiles_5.append(col.quantile(0.05))
            percentiles_95.append(col.quantile(0.95))
            missing_counts_num.append(col.isna().sum())

        elif col.dropna().apply(lambda x: isinstance(x, (str, list))).all():
            labels_cat.append(col_name)
            non_null = col.dropna()
            if non_null.apply(lambda x: isinstance(x, list)).all():
                non_null = non_null.explode()
            unique_counts.append(non_null.nunique())
            missing_counts_cat.append(col.isna().sum())
            proportions.append((non_null.value_counts(normalize=True) * 100).round(2))

    output_dir = Path('stats_tables')
    output_dir.mkdir(parents=True, exist_ok=True)

    df_num = pd.DataFrame({
        'Kolumna': labels_num,
        'Średnia': means,
        'Mediana': medians,
        'Min': min_vals,
        'Max': max_vals,
        'Odch_std': std_devs,
        'Percentyl_5': percentiles_5,
        'Percentyl_95': percentiles_95,
        'Braki': missing_counts_num
    })

    df_num.to_csv(output_dir / 'statystyki_liczbowe.csv', index=False)

    df_cat = pd.DataFrame({
        'Kolumna': labels_cat,
        'Unikalnych klas': unique_counts,
        'Braki': missing_counts_cat
    })

    df_cat.to_csv(output_dir / 'statystyki_kategoryczne.csv', index=False)

    for name, prop in zip(labels_cat, proportions):
        filename = output_dir / f'proporcje_{name}.csv'
        prop.to_csv(filename, header=['Procent'])

    print('Stats saved to CSV file')

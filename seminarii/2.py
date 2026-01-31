import pandas as pd
from functii import nan_replace_df, calcul_procent, f_disim, f_entropy

pd.set_option("display.max_columns", None)

# --- 1. Incarcarea si Curatarea Datelor Initiale (Nivel Localitate) ---
# 'index_col=0' seteaza prima coloana (probabil codul localitatii) ca index
df_etnii_localitati = pd.read_csv("data_in/Ethnicity.csv", index_col=0)

# Extrage numele coloanelor care contin datele despre etnii
coloane_etnii = list(df_etnii_localitati.columns)[1:]

nan_replace_df(df_etnii_localitati)

# --- 2. Agregarea Datelor la Nivel de Judet ---
# Incarca fisierul de mapare (cod localitate -> nume Judet)
df_coduri_localitati = pd.read_csv("data_in/Coduri_Localitati.csv", index_col=0)

# Combina datele despre etnii cu informatiile despre judet
# Se selecteaza doar coloanele cu etnii si se adauga coloana "County"
# Imbinarea (merge) se face pe baza indexului (codul localitatii)
df_etnii_cu_judet = df_etnii_localitati[coloane_etnii].merge(
    df_coduri_localitati["County"], left_index=True, right_index=True
)

# Grupeaza datele dupa coloana "County" si insumeaza valorile
# index-ul devine numele judetului
df_agregat_judet = df_etnii_cu_judet.groupby(by="County").sum()

# Salveaza datele agregate la nivel de judet intr-un fisier CSV
df_agregat_judet.to_csv("data_out/Ethnicity_County.csv")

# --- 3. Agregarea Datelor la Nivel de Regiune ---

# Incarca fisierul de mapare (nume Judet -> nume Regiune)
df_coduri_judete = pd.read_csv("data_in/Coduri_Judete.csv", index_col=0)


# Combina datele agregate pe judet cu informatiile despre regiune
# Imbinarea se face pe baza indexului (numele judetului)
df_judet_cu_regiune = df_agregat_judet.merge(
    df_coduri_judete["Regiune"], left_index=True, right_index=True
)

# Grupeaza datele dupa coloana "Regiune" si insumeaza valorile
# pentru a obtine totalul populatiei pe etnii la nivel de regiune
# index-ul devine numele regiunii
df_agregat_regiune = df_judet_cu_regiune.groupby(by="Regiune").sum()

# Salveaza datele agregate la nivel de regiune intr-un fisier CSV
df_agregat_regiune.to_csv("data_out/Ethnicity_Region.csv")

# Comentariu pentru o posibila extindere viitoare

# --- 4. Calculul Procentelor (Structura Etnica) ---

# Calculeaza structura procentuala a etniilor pentru fiecare localitate
# 'axis=1' aplica functia 'calcul_procent' pe fiecare linie (localitate)
df_procent_localitate = df_etnii_localitati[coloane_etnii].apply(
    func=calcul_procent, axis=1
)
df_procent_localitate.to_csv("data_out/Ethnicity_loc_p.csv")

# Calculeaza structura procentuala a etniilor pentru fiecare judet
# 'axis=1' aplica functia lambda pe fiecare linie (judet)
# Functia lambda este identica cu 'calcul_procent' (logica originala pastrata)
df_procent_judet = df_agregat_judet.apply(func=lambda x: x * 100 / x.sum(), axis=1)
df_procent_judet.to_csv("data_out/Ethnicity_county_p.csv")


# --- 5. Calculul Indicilor de Segregare (Nivel Judet) ---

# Calculeaza Indexul de Disimilaritate (f_disim) pentru fiecare judet
# Se grupeaza datele pe judet, iar functia 'f_disim' este aplicata
# pe sub-DataFrame-ul corespunzator fiecarei grupe (judet)
# 'include_groups=False' este folosit pentru a mentine logica originala a 'apply'
df_disim_judet = df_etnii_cu_judet.groupby(by="County").apply(
    func=f_disim, include_groups=False
)

df_disim_judet.to_csv("data_out/Dissim_County.csv")

# Calculeaza Indexul de Entropie (f_entropy) pentru fiecare judet
# Se aplica aceeasi logica de grupare ca la disimilaritate
df_entropie_judet = df_etnii_cu_judet.groupby(by="County").apply(
    func=f_entropy, include_groups=False
)

df_entropie_judet.to_csv("data_out/Entropy_County.csv")

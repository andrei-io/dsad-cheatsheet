import sys
import numpy as np
import pandas as pd
import factor_analyzer as fa
from geopandas import GeoDataFrame
from functii import nan_replace_df, salvare_ndarray, tabelare_varianta_factori
from grafice import plot_varianta, corelograma, plot_scoruri_corelatii, plot_harta, show

# --- 1. Configurarea Optiunilor si Incarcarea Datelor ---
pd.set_option("display.max_columns", None)
np.set_printoptions(precision=3, threshold=sys.maxsize, suppress=True)

# Incarcam setul de date teritorial
df_date = pd.read_csv("data_in/Teritorial2022/Teritorial_2022.csv", index_col=0)
nan_replace_df(df_date)

# Identificam variabilele numerice pentru analiza
nume_variabile = list(df_date.columns)[3:]
matrice_date = df_date[nume_variabile].values
n, m = matrice_date.shape

# --- 2. Testarea Factorabilitatii (Validarea Modelului) ---
# Inainte de AF, verificam daca variabilele sunt suficient corelate intre ele.

# 2.1. Testul Bartlett de Sfericitate
# p-value < 0.001 indica faptul ca matricea de corelatie nu este una identitate.
chi_sq, p_val_bartlett = fa.calculate_bartlett_sphericity(matrice_date)
if p_val_bartlett > 0.001:
    print("Eroare: Datele nu prezinta structura factoriala!")
    exit(0)

# 2.2. Indexul KMO (Kaiser-Meyer-Olkin)
# Masoara cat de adecvate sunt datele; valori > 0.6 sunt considerate bune.
kmo_pe_variabile, kmo_total = fa.calculate_kmo(matrice_date)
df_kmo = pd.DataFrame({"KMO": np.append(kmo_pe_variabile, kmo_total)},
                     index=nume_variabile + ["Total"])
df_kmo.to_csv("data_out_fa/kmo.csv")
corelograma(df_kmo, "Index KMO", vmin=0, cmap="Reds")

# --- 3. Constructia Modelului AF cu Rotatie Varimax ---
# Rotația 'varimax' maximizeaza varianta incarcarilor pentru a face factorii mai usor de interpretat.
model_af = fa.FactorAnalyzer(n_factors=m, rotation="varimax")
model_af.fit(matrice_date)

# --- 4. Analiza Variantei Factorilor Comuni ---
varianta_raw = model_af.get_factor_variance()
df_varianta = tabelare_varianta_factori(varianta_raw)
df_varianta.round(3).to_csv("data_out_fa/Varianta.csv")

# Determinam numarul relevant de factori folosind cele 3 criterii (Kaiser, Acoperire, Cattell).
k1, k2, k3 = plot_varianta(varianta_raw[0], 70, eticheta_x="Factor", titlu="Scree Plot Factori")
nr_factori_semnificativi = min([v for v in [k1, k2, k3] if v is not None])

# --- 5. Analiza Comunalitatilor si a Incarcarilor Factoriale ---

# 5.1. Comunalitatile si Varianta Specifica
# Comunalitatea indica varianta partajata; Uniqueness indica varianta reziduala.
comm = model_af.get_communalities()
psi = model_af.get_uniquenesses()
df_calitate = pd.DataFrame({"Comunalitati": comm, "Varianta specifica": psi}, index=nume_variabile)
df_calitate.to_csv("data_out_fa/Calitate_Variabile.csv")
corelograma(df_calitate, "Calitate Variabile (Comm si Psi)", vmin=0, cmap="YlGn")

# 5.2. Incarcarile Factoriale (Loadings)
# Reprezinta corelatia dintre variabilele originale si noii factori comuni.
df_loadings = salvare_ndarray(model_af.loadings_, nume_variabile, df_varianta.index,
                              "Indicatori", "data_out_fa/l.csv")

# Vizualizam incarcarile pentru primii doi factori pe cercul corelatiilor.
plot_scoruri_corelatii(df_loadings, "F1", "F2", "Cercul Corelatiilor AF",
                       etichete=df_loadings.index, corelatii=True)

# --- 6. Analiza Scorurilor Factoriale si Distributia Geografica ---

# Calculam scorurile (coordonatele localitatilor in noul spatiu).
scoruri = model_af.transform(matrice_date)
df_scoruri = salvare_ndarray(scoruri, df_date.index, df_varianta.index,
                            df_date.index.name, "data_out_fa/f.csv")

# Afisam distributia primilor factori pe harta Romaniei.
gdf = GeoDataFrame.from_file("data_in/RO_NUTS2/Ro.shp")
for i in range(1, nr_factori_semnificativi + 1):
    plot_harta(gdf, "sj", df_scoruri, "F" + str(i))

show()
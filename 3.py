import sys
import numpy as np
import pandas as pd

from functii import nan_replace_df, acp, tabelare_varianta, salvare_ndarray
from grafice import plot_varianta, show, corelograma, plot_scoruri_corelatii

# --- Configurarea Optiunilor de Afisare ---

pd.set_option("display.max_columns", None)
np.set_printoptions(3, sys.maxsize, suppress=True)

# --- 1. Incarcarea si Pregatirea Datelor ---
df_date = pd.read_csv("data_in/Teritorial2022/Teritorial_2022.csv", index_col=0)
nan_replace_df(df_date)

# Extrage numele coloanelor care contin variabilele de analizat
nume_variabile_observate = list(df_date.columns)[3:]

# Extrage datele numerice intr-o matrice NumPy
matrice_date = df_date[nume_variabile_observate].values

# --- 2. Aplicarea ACP (Analiza Componentelor Principale) ---

# Apeleaza functia ACP (logica originala presupune standardizare 'scal=True')
# matrice_prelucrata: Datele centrate si standardizate (fost 'x_')
# matrice_cov_cor: Matricea de corelatie (daca scal=True) (fost 'r_v')
# valori_proprii: Valorile proprii sortate descrescator (fost 'alpha')
# vectori_proprii: Vectorii proprii sortati (fost 'a')
matrice_prelucrata, matrice_cov_cor, valori_proprii, vectori_proprii = acp(matrice_date)

# --- 3. Analiza Matricei de Corelatie ---
df_matrice_corelatie = salvare_ndarray(
    matrice_cov_cor,
    nume_variabile_observate,
    nume_variabile_observate,
    "Indicatori",
    "data_out/R.csv",
)
# 'annot=True' doar daca sunt mai putin de 10 variabile (logica originala)
corelograma(df_matrice_corelatie, annot=len(nume_variabile_observate) < 10)

# --- 4. Analiza Variantei (Valorilor Proprii) ---

tabel_varianta = tabelare_varianta(valori_proprii)
tabel_varianta.round(3).to_csv("data_out/Varianta.csv")

# Genereaza "Scree Plot" (plotul variantei) si retine numarul
# de componente sugerat de cele trei criterii
k1_kaiser, k2_acoperire, k3_cattell = plot_varianta(valori_proprii)

# --- 5. Calculul Componentelor si Corelatiilor Factoriale ---

# Calculeaza componentele principale (scorurile) (C = X_prelucrat * A)
# (fost 'c')
componente_principale = matrice_prelucrata @ vectori_proprii

# Obtine dimensiunile matricei de date initiale
nr_linii, nr_coloane = matrice_date.shape

# Analiza corelatiilor dintre variabilele observate (X) si componente (C)
# Se calculeaza matricea de corelatie intre X_prelucrat si C
# si se extrage sub-matricea corecta (fost 'r_xc')
matrice_corelatii_var_comp = np.corrcoef(
    matrice_prelucrata, componente_principale, rowvar=False
)[:nr_coloane, nr_coloane:]

# Salveaza matricea de corelatii factoriale (R_XC) in CSV
df_corelatii_var_comp = salvare_ndarray(
    matrice_corelatii_var_comp,
    nume_variabile_observate,
    tabel_varianta.index,  # Foloseste C1, C2... ca nume de coloane
    "Indicatori",
    "data_out/R_XC.csv",
)

corelograma(
    df_corelatii_var_comp,
    "Corelograma corelatii factoriale",
    annot=nr_coloane < 10,  # 'annot=True' daca sunt putine variabile
)

# --- 6. Analiza Scorurilor ---

# Calculeaza scorurile standardizate (fost 's')
scoruri_standardizate = componente_principale / np.sqrt(valori_proprii)

# Salveaza componentele principale (nestandardizate) (fost 't_c')
df_componente_principale = salvare_ndarray(
    componente_principale,
    df_date.index,  # Indexul original (ex: nume localitati)
    df_corelatii_var_comp.columns,  # Numele componentelor (C1, C2...)
    df_date.index.name,
    "data_out/C.csv",
)

# Salveaza scorurile standardizate (fost 't_s')
df_scoruri_standardizate = salvare_ndarray(
    scoruri_standardizate,
    df_date.index,
    df_corelatii_var_comp.columns,
    df_date.index.name,
    "data_out/S.csv",
)

plot_scoruri_corelatii(df_componente_principale, titlu="Plot componente")

# Afiseaza toate graficele generate (Scree Plot, Corelograme, Plot Scoruri)
# show()

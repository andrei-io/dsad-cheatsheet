import sys
import numpy as np
import pandas as pd

from functii import nan_replace, teste_concordanta

# --- Configurarea Optiunilor de Afisare ---

pd.set_option("display.max_columns", None)

np.set_printoptions(
    precision=3,  # Afiseaza numerele cu 3 zecimale
    threshold=sys.maxsize,  # Asigura afisarea completa a array-urilor (fara "...")
    suppress=True,  # Opreste notatia stiintifica (ex: 1.23e+4)
)

# 'index_col=0' specifica faptul ca prima coloana din CSV este folosita ca index
tabel_date_df = pd.read_csv("data_in/Teritorial_2022.csv", index_col=0)

# Obtine o lista cu numele coloanelor care contin variabilele numerice
# Se presupune ca acestea incep de la a patra coloana (index 3) pana la sfarsit
nume_coloane_numerice = list(tabel_date_df.columns)[3:]

# Extrage datele din coloanele numerice selectate
# si le converteste intr-o matrice NumPy (un array 2D)
# '.values' returneaza o reprezentare NumPy a datelor din DataFrame
date_numerice_np = tabel_date_df[nume_coloane_numerice].values

# --- Procesarea Datelor ---

# Apelarea functiei de inlocuire a valorilor lipsa (NaN)
nan_replace(date_numerice_np)

# Verificare (comentata) pentru a confirma ca nu mai exista valori NaN

# Aplicarea testelor de concordanta (pentru normalitate)
# Functia ruleaza testele (ex: Shapiro, KS) pe fiecare coloana a matricei
# NOTA: Rezultatul intors de functie nu este stocat intr-o variabila
#       si nici nu este afisat in acest script.
rezultate_teste = teste_concordanta(date_numerice_np)

# salveaza rezulate_teste in data_out/test.csv
pd.DataFrame(
    rezultate_teste,
    index=nume_coloane_numerice,
    columns=[
        "Shapiro > 0.1",
        "KS >= 0.05",
        "Chi2 < 0.01",
        "Alt test",
    ],
).to_csv("data_out/test.csv")

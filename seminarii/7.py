import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage
from functii import nan_replace_df, calcul_partitie
from grafice import plot_ierarhie, show

# --- 1. Încărcarea și Pregătirea Datelor ---
# Analizăm datele privind mortalitatea în România
t = pd.read_csv("data_in/MortalitateRO2019/mortalitate_ro.csv", index_col=1)
nan_replace_df(t)

# Selectăm variabilele de interes (excludem prima coloană dacă nu e numerică)
variabile_observate = list(t)[1:]
x = t[variabile_observate].values

# --- 2. Construirea Ierarhiei (Metoda Linkage) ---
# 'complete' linkage calculează distanța dintre clustere pe baza celor mai îndepărtate puncte.
metoda_grupare = "complete"
h = linkage(x, metoda_grupare)

print("Matricea ierarhie (primele 5 rânduri):")
print(h[:5])

# Vizualizarea dendrogramre initiale (fără secționare)
plot_ierarhie(h, t.index, titlu="Dendrograma Ierarhica - Metoda " + metoda_grupare)

# --- 3. Determinarea și Analiza Partiției Optimale ---
# Folosim metoda "Elbow" (cotului) pentru a găsi saltul maxim în distanțele de fuziune.
k_opt, threshold_opt, p_opt = calcul_partitie(h)

print(f"Număr clustere în partiția optimală: {k_opt}")
print(f"Distanța de secționare (threshold): {threshold_opt:.3f}")

# Re-plotăm ierarhia cu linia de secționare pentru partiția optimală
plot_ierarhie(h, t.index, threshold_opt, "Partitia Optimala (k=" + str(k_opt) + ")")

# Salvăm apartenența la clustere într-un DataFrame
t_partitii = pd.DataFrame(data={"Partitie_Optimala": p_opt}, index=t.index)

# --- 4. Analiza unei Partiții cu Număr Fix de Clustere ---
# Putem forța algoritmul să ne ofere un număr specific de grupuri (ex: k=3).
k_fix = 3
k, threshold_k, p_k = calcul_partitie(h, k=k_fix)

plot_ierarhie(h, t.index, threshold_k, f"Partitia cu {k_fix} Clusteri")
t_partitii[f"Partitie_{k_fix}"] = p_k

# --- 5. Exportul Rezultatelor ---
t_partitii.to_csv("data_out/Partitii_Mortalitate.csv")
print("Rezultatele au fost salvate în 'data_out/Partitii.csv'.")

show()

# 0. Comun la orice subiect
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing

# 1. Citim fisierul
df = pd.read_csv("DateleTale.csv", index_col=0)

# 2. Alegem coloanele cu numere
# Luam toate numele coloanelor, mai putin prima (care e text/tara)
cols = df.columns[1:]

# 3. Umplem golurile (NaN) cu media direct in tabel
df[cols] = df[cols].fillna(df[cols].mean())

# 4. Facem X (Matricea de numere) si X_std (Standardizat)
X = df[cols].values

scaler = sklearn.preprocessing.StandardScaler()
X_std = scaler.fit_transform(X)

# *VARIANTA 1: CLUSTERIZARE*

import scipy.cluster.hierarchy as sch
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA

# --- A. Matricea Ierarhie si Dendrograma ---
matrice = sch.linkage(X_std, method='ward')
print("Matricea:\n", matrice)

sch.dendrogram(matrice, labels=df.index)
plt.title("Dendrograma")
plt.show()

# --- B. Calcul k_optim (Elbow pe distante) ---
dist_agregare = matrice[:, 2]
difere_dist = np.diff(dist_agregare)
k_optim = len(dist_agregare) - np.argmax(difere_dist)
print("Numar optim clusteri (Elbow):", k_optim)

# --- C. Partitie k-ales (Schimba k aici daca vrei altceva) ---
k_ales = 3
labels_k = sch.fcluster(matrice, t=k_ales, criterion='maxclust')
df['Cluster'] = labels_k

# --- D. Indecsi Silhouette (Mediu si per instanta) ---
sil_avg = silhouette_score(X_std, labels_k)
sil_inst = silhouette_samples(X_std, labels_k)
print("Silhouette Mediu:", sil_avg)

# --- E. Plot Silhouette (Grafic) ---
plt.figure()
y_start = 10
for i in range(1, k_ales + 1):
    val_c = sil_inst[labels_k == i]
    val_c.sort()
    plt.fill_betweenx(np.arange(y_start, y_start + len(val_c)), 0, val_c)
    y_start += len(val_c) + 10
plt.axvline(sil_avg, color='red', linestyle='--')
plt.title(f"Silhouette pentru k={k_ales}")
plt.show()

# --- F. Histograme variabile pe clusteri ---
for col in cols:
    plt.figure()
    for i in range(1, k_ales + 1):
        subset = df[df['Cluster'] == i]
        plt.hist(subset[col], alpha=0.5, label=f"C{i}")
    plt.title(f"Distributie: {col}")
    plt.legend()
    plt.show()

# --- G. Plot in axe principale (PCA) ---
pca_model = PCA(n_components=2)
componente = pca_model.fit_transform(X_std)
plt.figure()
plt.scatter(componente[:, 0], componente[:, 1], c=labels_k, cmap='viridis')
plt.title("Clusteri in axe PCA")
plt.show()

# --- H. Salvare ---
df.to_csv("Rezultat_Cluster.csv")

# *VARIANTA 2: PCA (Componente Principale)*
# *Dacă cere: Varianța, Cercul, Scoruri.*


# *VARIANTA 2: PCA (Componente Principale)*
from sklearn.decomposition import PCA
import seaborn as sns

# --- A. Modelul PCA si Varianta ---
pca = PCA()
C = pca.fit_transform(X_std) # Scorurile (Componentele)
alpha = pca.explained_variance_ # Valorile proprii (Eigenvalues)
proportie = pca.explained_variance_ratio_ # Varianta explicata (%)

# SALVARE SCORURI
df_scoruri = pd.DataFrame(C, index=df.index, columns=[f"PC{i+1}" for i in range(C.shape[1])])
df_scoruri.to_csv("scoruri.csv")

print("Varianta axelor (Eigenvalues):", alpha)

# --- B. Plot Varianta cu Criterii (Scree Plot) ---
plt.figure()
plt.plot(range(1, len(alpha) + 1), alpha, 'ro-')
plt.axhline(y=1, color='b', linestyle='--', label='Criteriul Kaiser')
plt.title("Scree Plot (Varianta Componentelor)")
plt.legend()
plt.show()

# --- C. Corelatii Factoriale (Variabile - Componente) ---
# Corelatia = loadings * sqrt(eigenvalues)
loadings = pca.components_.T
corelatii = loadings * np.sqrt(alpha)
df_corelatii = pd.DataFrame(corelatii, index=cols, columns=[f"PC{i+1}" for i in range(len(alpha))])

# --- D. Corelograma Corelatii ---
plt.figure(figsize=(10, 8))
sns.heatmap(df_corelatii, annot=True, cmap='coolwarm')
plt.title("Corelograma Corelatiilor Factoriale")
plt.show()

# --- E. Cercul Corelatiilor ---
plt.figure(figsize=(8, 8))
plt.scatter(corelatii[:, 0], corelatii[:, 1])
for i in range(len(cols)):
    plt.arrow(0, 0, corelatii[i, 0], corelatii[i, 1], color='r', alpha=0.5)
    plt.text(corelatii[i, 0], corelatii[i, 1], cols[i])
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.title("Cercul Corelatiilor")
plt.show()

# --- F. Plot Componente / Scoruri ---
plt.figure()
plt.scatter(C[:, 0], C[:, 1], c='blue')
for i, txt in enumerate(df.index):
    plt.text(C[i, 0], C[i, 1], txt, fontsize=8)
plt.title("Plot Scoruri (Componente)")
plt.show()

# --- G. Cosinusuri si Contributii ---
# Cosinusuri (Calitatea reprezentarii punctelor pe axe)
C2 = C**2
sum_C2 = np.sum(C2, axis=1, keepdims=True)
cosin = C2 / sum_C2
print("Cosinusuri (primele 5 randuri):\n", cosin[:5])

# Contributii (Cat de mult influenteaza fiecare observatie axa)
contrib = (C2 / (alpha * len(df))) * 100
print("Contributii (%):\n", contrib[:5])

# --- H. Comunalitati si Corelograma Comunalitati ---
# Comunalitatea = suma patratelor corelatiilor pe axele retinute
# Aici calculam pentru toate axele
comunalitati = np.cumsum(corelatii**2, axis=1)
df_comun = pd.DataFrame(comunalitati, index=cols, columns=[f"PC1-PC{i+1}" for i in range(len(alpha))])

plt.figure(figsize=(10, 6))
sns.heatmap(df_comun, annot=True, cmap='Greens')
plt.title("Corelograma Comunalitatilor")
plt.show()

# *VARIANTA 3: ANALIZA FACTORIALA (FA)*

from factor_analyzer import FactorAnalyzer, calculate_bartlett_sphericity, calculate_kmo
import seaborn as sns

# --- A. Teste Factorabilitate (Bartlett si KMO) ---
chi_sq, p_val = calculate_bartlett_sphericity(X_std)
kmo_all, kmo_model = calculate_kmo(X_std)
print(f"Bartlett p-value: {p_val} (Trebuie < 0.05)")
print(f"Scor KMO: {kmo_model} (Trebuie > 0.6)")

# --- B. Determinare numar factori (Kaiser) ---
fa_temp = FactorAnalyzer(rotation=None)
fa_temp.fit(X_std)
ev, _ = fa_temp.get_eigenvalues()
k_fa = sum(ev > 1)
print(f"Numar factori (Kaiser): {k_fa}")

# --- C. Modelul Final (cu rotatie Varimax sau None) ---
# Schimba rotation='varimax' in rotation=None daca se cere fara rotatie
fa_model = FactorAnalyzer(n_factors=k_fa, rotation='varimax')
fa_model.fit(X_std)

# --- D. Varianta Factorilor (proportie, cumulata) ---
# [0] - Varianta, [1] - Proportie, [2] - Cumulata
var_info = fa_model.get_factor_variance()
print("Varianta Factori:\n", var_info)

# --- E. Corelatii Factoriale (Loadings) si Corelograma ---
loadings_fa = fa_model.loadings_
df_loadings = pd.DataFrame(loadings_fa, index=cols, columns=[f"F{i+1}" for i in range(k_fa)])

plt.figure(figsize=(10, 8))
sns.heatmap(df_loadings, annot=True, cmap='RdBu')
plt.title("Corelograma Corelatiilor Factoriale (Loadings)")
plt.show()

# --- F. Cercul Corelatiilor ---
plt.figure(figsize=(8, 8))
plt.scatter(loadings_fa[:, 0], loadings_fa[:, 1])
for i in range(len(cols)):
    plt.arrow(0, 0, loadings_fa[i, 0], loadings_fa[i, 1], color='r', alpha=0.4)
    plt.text(loadings_fa[i, 0], loadings_fa[i, 1], cols[i])
plt.xlabel("Factor 1"); plt.ylabel("Factor 2")
plt.title("Cercul Corelatiilor (FA)")
plt.show()

# --- G. Comunalitati si Varianta Specifica ---
comunalit_fa = fa_model.get_communalities()
var_specifica = fa_model.get_uniquenesses()
df_comun_spec = pd.DataFrame({"Comunalitati": comunalit_fa, "Varianta Specifica": var_specifica}, index=cols)

plt.figure(figsize=(8, 6))
sns.heatmap(df_comun_spec, annot=True, cmap='YlGn')
plt.title("Corelograma Comunalitati si Varianta Specifica")
plt.show()

# --- H. Scoruri Factoriale si Plot ---
scoruri_fa = fa_model.transform(X_std)
df_scoruri_fa = pd.DataFrame(scoruri_fa, index=df.index, columns=[f"F{i+1}" for i in range(k_fa)])
df_scoruri_fa.to_csv("scoruri_fa.csv")

plt.figure()
plt.scatter(scoruri_fa[:, 0], scoruri_fa[:, 1], c='green')
for i, txt in enumerate(df.index):
    plt.text(scoruri_fa[i, 0], scoruri_fa[i, 1], txt, fontsize=8)
plt.title("Plot Scoruri Factoriale")
plt.show()


# *VARIANTA 4: DISCRIMINANTA (LDA)*
# *Dacă cere: Predicție, Matrice Confuzie (Ti se da o anumita coloana!)*


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix

# A. Definim Tinta (Ce vrem sa ghicim)
# Inlocuieste 'Continent' cu ce iti cere domnul felix.
y = df['Continent'].values

# B. Antrenam modelul
lda = LinearDiscriminantAnalysis()
lda.fit(X_std, y)

# C. Facem Predicția
preziceri = lda.predict(X_std)

# D. Verificam (Matricea de confuzie)
print(confusion_matrix(y, preziceri))

# E. Salvam
df['Prezicere'] = preziceri
df.to_csv("Rezultat_LDA.csv")
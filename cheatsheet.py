
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
print("Matricea ierarhică:\n", matrice)

sch.dendrogram(matrice, labels=df.index)
plt.title("Dendrograma (Ward)")
plt.show()

# --- B. Calcul k_optim (Elbow pe distante) ---
distante = matrice[:, 2]
diff_dist = np.diff(distante)
idx_max = np.argmax(diff_dist)

prag_opt = distante[idx_max]
print("Prag optim:", prag_opt)

plt.figure(figsize=(10, 5))
sch.dendrogram(
    matrice,
    labels=df.index,
    color_threshold=prag_opt
)
plt.axhline(y=prag_opt, color='red', linestyle='--', label='Prag optim')
plt.legend()
plt.title("Dendrograma – partiția optimă")
plt.show()

# --- C. Partitie k-ales (Schimba k aici daca vrei altceva) ---
# NE UITAM PE ULTIMA DENDOGRAMA
k_ales = 4
prag_k=matrice[-(k_ales-1),2]
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



# *VARIANTA 4: ANALIZA DISCRIMINANTA (LDA)*
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# --- A. Pregatire Date si Split (Antrenare/Testare) ---
# Tinta: coloana categorica (ex: 'DECISION' sau 'Continent')
y = df['DECISION'].values
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.3, random_state=42)

# --- B. Model Liniar si Scoruri Discriminante (Z) ---
lda_lin = LDA()
lda_lin.fit(X_train, y_train)

# Calcul scoruri pentru tot setul original (pentru z.csv)
z_scores = lda_lin.transform(X_std)
df_z = pd.DataFrame(z_scores, index=df.index, columns=[f"Z{i+1}" for i in range(z_scores.shape[1])])
df_z.to_csv("./data_out/z.csv")

# --- C. Graficul Scorurilor (Axe Discriminante) ---
plt.figure(figsize=(8, 6))
# Daca ai 3 clase, vei avea automat 2 axe (Z1 si Z2)
plt.scatter(z_scores[:, 0], z_scores[:, 1], c=pd.factorize(y)[0], cmap='viridis')
plt.xlabel("Z1"); plt.ylabel("Z2")
plt.title("Graficul scorurilor discriminante")
plt.show()

# --- D. Evaluare Model (Matrice Confuzie si Acuratete) ---
pred_test = lda_lin.predict(X_test)

# Matricea de confuzie
clase = lda_lin.classes_
m_conf = confusion_matrix(y_test, pred_test)
df_matc = pd.DataFrame(m_conf, index=clase, columns=clase)
df_matc.to_csv("./data_out/matc.csv")

# Indicatori la consola
print("Acuratete Globala:", accuracy_score(y_test, pred_test))
print("Raport Clasificare:\n", classification_report(y_test, pred_test))

# --- E. Predictia pe Setul de Aplicare (Pacienti_apply.csv) ---
df_apply = pd.read_csv("./data_in/Pacienti_apply.csv", index_col=0)
# Selectam aceleasi coloane si standardizam cu scaler-ul initial
X_apply = df_apply[cols].values
X_apply_std = scaler.transform(X_apply)

# Facem predictia si salvam
pred_apply = lda_lin.predict(X_apply_std)
df_apply['PREDICTIE'] = pred_apply
df_apply.to_csv("./data_out/Pacienti_results.csv")

# --- F. Vizualizarea Distributiilor pe prima axa ---
plt.figure()
for clasa in clase:
    subset = z_scores[y == clasa]
    plt.hist(subset[:, 0], alpha=0.5, label=clasa)
plt.title("Distributia pe axa Z1")
plt.legend()
plt.show()

# *VARIANTA 5: ANALIZA CORELATIILOR CANONICE (CCA)*

from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler

# --- A. Pregatire seturi de variabile (X si Y) ---
# Exemplu: X sunt indicatori economici, Y sunt indicatori sociali
cols_x = ['Ind1', 'Ind2'] # Inlocuieste cu coloanele tale
cols_y = ['Ind3', 'Ind4']
X_set = df[cols_x].values
Y_set = df[cols_y].values

# Standardizam ambele seturi
X_s = StandardScaler().fit_transform(X_set)
Y_s = StandardScaler().fit_transform(Y_set)

# --- B. Modelul CCA si Scoruri Canonice ---
n_comp = min(X_s.shape[1], Y_s.shape[1])
cca = CCA(n_components=n_comp)
Z, U = cca.fit_transform(X_s, Y_s)
# U - scorurile pentru setul X, V - scorurile pentru setul Y

# --- C. Corelatii Canonice (intre perechile U si V) ---
cor_can = np.array([np.corrcoef(Z[:, i], U[:, i])[0, 1] for i in range(n_comp)])
print("Corelatii canonice (r):", cor_can)

# --- D. Relevanta radacinilor (Test Bartlett adaptat) ---
def test_bartlett(r, n):
    p = X_s.shape[1]
    q = Y_s.shape[1]
    m = n - 1 - 0.5 * (p + q + 1)
    chi_sq = -m * np.log(np.prod(1 - r**2))
    df_bartlett = p * q
    return chi_sq, df_bartlett

chi_val, df_val = test_bartlett(cor_can, len(df))
print(f"Test Bartlett: Chi_sq={chi_val}, df={df_val}")

# --- E. Corelatii variabile observate - variabile canonice (Structura) ---
# Corelatii X cu U
cor_x_u = np.corrcoef(X_s, Z, rowvar=False)[:X_s.shape[1], X_s.shape[1]:]
# Corelatii Y cu V
cor_y_v = np.corrcoef(Y_s, U, rowvar=False)[:Y_s.shape[1], Y_s.shape[1]:]

# --- F. Cercul Corelatiilor (Biplot Corelatii) ---
plt.figure(figsize=(8, 8))
plt.scatter(cor_x_u[:, 0], cor_x_u[:, 1], color='r', label='Set X')
plt.scatter(cor_y_v[:, 0], cor_y_v[:, 1], color='b', label='Set Y')
for i, txt in enumerate(cols_x): plt.text(cor_x_u[i,0], cor_x_u[i,1], txt)
for i, txt in enumerate(cols_y): plt.text(cor_y_v[i,0], cor_y_v[i,1], txt)
plt.title("Cercul Corelatiilor Canonice")
plt.legend(); plt.show()

# --- G. Corelograma Corelatii ---
df_cor_xu = pd.DataFrame(cor_x_u, index=cols_x, columns=[f"U{i+1}" for i in range(n_comp)])
sns.heatmap(df_cor_xu, annot=True, cmap='RdBu')
plt.title("Corelatii Variabile X - Scoruri U")
plt.show()

# --- H. Plot Instante (Biplot Scoruri U1 vs V1) ---
plt.figure()
plt.scatter(Z[:, 0], U[:, 0])
plt.xlabel("U1 (Set X)"); plt.ylabel("V1 (Set Y)")
plt.title("Instante in spatiul primei perechi canonice")
plt.show()

# --- I. Plot Instante – primele doua radacini canonice ---
plt.figure(figsize=(8, 6))

# Set Barbati (Z)
plt.scatter(Z[:, 0], Z[:, 1], marker='o', label='Barbati (Z)')

# Set Femei (U)
plt.scatter(U[:, 0], U[:, 1], marker='x', label='Femei (U)')

plt.xlabel("Radacina canonica 1")
plt.ylabel("Radacina canonica 2")
plt.title("Plot instante – primele doua radacini canonice")
plt.legend()
plt.grid(True)
plt.show()

# --- I. Varianta Explicata si Redundanta ---
# Varianta explicata de variabilele canonice pentru setul lor
var_expl_x = np.mean(cor_x_u**2, axis=0)
var_expl_y = np.mean(cor_y_v**2, axis=0)
# Redundanta (cata varianta din Y e explicata de U)
redundanta_y = var_expl_y * (cor_can**2)
print("Varianta explicata X:", var_expl_x)
print("Redundanta Y (prin X):", redundanta_y)
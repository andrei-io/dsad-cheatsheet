
# 0. Comun la orice subiect
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 1. Citim fisierul
df = pd.read_csv("DateleTale.csv", index_col=0)

# 2. Alegem coloanele cu numere
# Luam toate numele coloanelor, mai putin prima (care e text/tara)
cols = df.columns[1:]

# 3. Umplem golurile (NaN) cu media direct in tabel
df[cols] = df[cols].fillna(df[cols].mean())

# 4. Facem X (Matricea de numere) si X_std (Standardizat)
X = df[cols].values

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# *VARIANTA 1: CLUSTERIZARE*
# *Dacă cere: Matrice, Dendrogramă, Partiție.*


import scipy.cluster.hierarchy as sch

# A. Matricea
# "ward" e metoda standard ceruta
matrice = sch.linkage(X_std, method='ward')
print(matrice)

# B. Dendrograma (Graficul)
sch.dendrogram(matrice)
plt.show()

# C. Facem Grupele (Clusterii)
k = 3 #cel mai probabil
labels = sch.fcluster(matrice, t=k, criterion='maxclust')

# D. Salvam in fisier
df['Cluster'] = labels
df.to_csv("Rezultat_Cluster.csv")


# *VARIANTA 2: PCA (Componente Principale)*
# *Dacă cere: Varianța, Cercul, Scoruri.*


from sklearn.decomposition import PCA

# A. Facem Modelul
pca = PCA()
pca.fit(X_std)

# B. Varianta (Cat de importante sunt axele)
print(pca.explained_variance_ratio_)

# C. Graficul (Cercul Corelatiilor)
# Luam coordonatele (Loadings)
loadings = pca.components_.T
plt.scatter(loadings[:, 0], loadings[:, 1])

# Punem numele coloanelor pe grafic
for i in range(len(cols)):
    plt.text(loadings[i, 0], loadings[i, 1], cols[i])

plt.show()

# *VARIANTA 3: FACTORIALĂ (Factor Analysis)*
# *Dacă cere: Kaiser, Varimax, Loadings.*


from factor_analyzer import FactorAnalyzer

# A. Aflam cati factori sa folosim (Kaiser)
# Calculam valorile proprii
ev, _ = FactorAnalyzer().fit(X_std).get_eigenvalues()
# Numaram cate sunt mai mari ca 1
nr_factori = sum(ev > 1)

# B. Rulam analiza finala (cu rotatie Varimax)
fa = FactorAnalyzer(n_factors=nr_factori, rotation='varimax')
fa.fit(X_std)

# C. Afisam Varianta
print(fa.get_factor_variance())

# D. Graficul (Fix ca la PCA)
loadings = fa.loadings_
plt.scatter(loadings[:, 0], loadings[:, 1])

for i in range(len(cols)):
    plt.text(loadings[i, 0], loadings[i, 1], cols[i])

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
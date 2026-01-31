# Examen DSAD - Nota 10 (sau 9 ca nu e C)

# =>Analiza Componentelor Principale (PCA)

#Importuri necesare:
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sb

#citim datele
date = pd.read_csv("date_pca.csv")
print(date)

date_numerice = date.drop(columns={"Tara"}) # pentru pca avem nevoie doar de coloanele numerice
print(date_numerice)

#standardizam datele
scaler = StandardScaler()
date_standard = scaler.fit_transform(date_numerice)

#aplicam PCA
pca = PCA()
date_pca = pca.fit(date_standard)
componente_principale = pca.components_

#Varianta componente
varianta = pca.explained_variance_
print("1.Varianta componentelor:\n", varianta)

#Plot varianta componente cu evidentierea criteriilor de relevanta
varianta_explicata = pca.explained_variance_ratio_
varianta_cumulativa = np.cumsum(varianta_explicata)
plt.figure(figsize=(8,6))
plt.bar(range(1, len(varianta)+1), varianta_explicata, alpha=0.7, color="blue", label="Varianta explicata")
plt.plot(range(1, len(varianta)+1), varianta_cumulativa, marker="o", linestyle="--", color="red", label="Varianta cumulativa")
plt.xlabel("Numarul componentelor principale")
plt.ylabel("Varianta explicata")
plt.title("Plot varianta componente")
plt.legend()
plt.grid()
plt.show()

# Calculul corelatii factoriale (corelatii variabile observate - componente)
# corelatie_factoriala = componente_principale.T * sqrt(varianta)
corelatie_factoriala = componente_principale.T * np.sqrt(varianta)
print("3. Corelatii factoriale:\n", corelatie_factoriala)

# Trasare corelograma corelatii factoriale
plt.figure(figsize=(10,8))
sb.heatmap(corelatie_factoriala, annot=True, cmap='coolwarm', center=0, xticklabels=[f"PC{i+1}" for i in range(0, corelatie_factoriala.shape[1])])
plt.title("Corelograma corelatiilor factoriale")
plt.xlabel("Componente principale")
plt.ylabel("Variabile observate")
plt.show()

# Trasare cercul corelatiilor
fig,ax = plt.subplots(figsize=(8,8))
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
circle = patches.Circle((0,0),1,color="gray", fill=False)
ax.add_patch(circle)

for i, (x,y) in enumerate(corelatie_factoriala[:, :2]):
    ax.arrow(0,0,x,y,head_width=0.05, head_length=0.05, fc="red", ec="red")
    ax.text(x,y,date_numerice.columns[i], fontsize=12, color="black")

ax.axhline(0,color="gray", linestyle="--")
ax.axvline(0, color="gray", linestyle="--")

plt.xlabel("Componenta principala 1")
plt.ylabel("Componenta principala 2")
plt.title("Cercul corelatiilor")
plt.grid()
plt.show()

# Calcul componente si/sau scoruri
scoruri_componente = pca.transform(date_standard)
print("Componente principale:\n", componente_principale)
print("Scoruri componente:\n", scoruri_componente)

# Trasare plot componente/scoruri
plt.figure(figsize=(8,6))
plt.scatter(scoruri_componente[:,0], scoruri_componente[:,1], c="blue", alpha=0.8)
plt.title("Plot componente/scoruri")
plt.xlabel("Componenta principala 1")
plt.ylabel("Componenta principala 2")
plt.grid()
plt.show()

# Calcul cosinusuri
# cos = componente^2 / suma pe fiecare variabila
componente_patrat = componente_principale**2
suma_var = np.sum(componente_patrat, axis=0)
cosinusuri = componente_patrat/suma_var
print("Cosinusuri:\n", cosinusuri)

# Calcul contributii
contributii = (pca.explained_variance_ratio_ * 100)
print("Contributii:\n", contributii)

# Calcul comunalitati
comunalitati = np.sum(corelatie_factoriala**2, axis=1)
print("Comunalitati:\n", comunalitati)

# Trasare corelograma comunalitati
plt.figure(figsize=(10,8))
sb.heatmap(pd.DataFrame(comunalitati, columns=['Comunalități'], index=date_numerice.columns),
            annot=True, cmap='viridis', fmt=".2f")
plt.title("Corelograma comunalitati")
plt.show()


# ------------------------------------------------------------------------------------------------------------------------------------
# => Analiza Factoriala (EFA)

# importuri
import pandas as pd
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
import seaborn as sb
import matplotlib.pyplot as plt

# citire csv-uri
date = pd.read_csv("factorial_analysis.csv")
print(date)
x = date.iloc[:,0:]
print(x)

# Analiza factorabilitatii - Bartlett
chi_square, p_value = calculate_bartlett_sphericity(x)
print("Factorabilitatea Barlett")
print(f"Valoarea chi-square: {chi_square}")
print(f"P-Value: {p_value}")

# Analiza factorabilitatii - KMO
kmo_all, kmo_model = calculate_kmo(x)
print(f"Scorul KMO total: {kmo_model}")

# Calcul varianta factori (cu/fara rotatie)
# fara rotatie
fa_fara_rotatie = FactorAnalyzer(n_factors=5, rotation=None)
fa_fara_rotatie.fit(x)
varianta_fara_rotatie = fa_fara_rotatie.get_factor_variance()
print(f"Varianta factori fara rotatie: {varianta_fara_rotatie}")

# cu rotatie
fa_cu_rotatie = FactorAnalyzer(n_factors=5, rotation="varimax")
fa_cu_rotatie.fit(x)
varianta_cu_rotatie = fa_cu_rotatie.get_factor_variance()
print(f"Varianta factori cu rotatie: {varianta_cu_rotatie}")

# Calcul corelatii factoriale (cu/fara rotatie)
# fara rotatie
coreletii_factoriale_fara = fa_fara_rotatie.loadings_
print(f"Corelatii factoriale fara rotatie: {coreletii_factoriale_fara}")

#cu rotatie
coreletii_factoriale_cu = fa_cu_rotatie.loadings_
print(f"Corelatii factoriale cu rotatie: {coreletii_factoriale_cu}")

# Trasare corelograma corelatii factoriale (cu/fara rotatie)
# fara rotatie
plt.figure(figsize=(10,8))
sb.heatmap(coreletii_factoriale_fara, center=0, cmap="coolwarm", annot=True, xticklabels=[f"{i+1}" for i in range(0,coreletii_factoriale_fara.shape[1])], yticklabels=x.columns)
plt.title("Corelograma factoriala fara rotatie")
plt.xlabel("Componente")
plt.ylabel("Variabile")
plt.show()

#cu rotatie
plt.figure(figsize=(10,8))
sb.heatmap(coreletii_factoriale_cu, center=0, cmap="coolwarm", annot=True, xticklabels=[f"{i+1}" for i in range(0, coreletii_factoriale_cu.shape[1])], yticklabels=x.columns)
plt.title("Corelograma factoriala cu rotatie")
plt.xlabel("Componente")
plt.ylabel("Variabile")
plt.show()

# Trasare cercul corelatiilor (cu/fara rotatie)
#fara rotatie
x1 = coreletii_factoriale_fara[:, 0] # corelatiile cu factorul1
y = coreletii_factoriale_fara[:, 1] # corelatiile cu factorul2

fig, ax = plt.subplots(figsize=(6,6))

circle = plt.Circle((0,0),1,color='b', fill=False, linestyle='dashed')
ax.add_patch(circle)
plt.scatter(x1,y)

for i, label in enumerate(x.columns):
    plt.text(x1[i], y[i], label, fontsize=9, ha='right', va='bottom')

plt.axhline(0,color='black', linewidth=0.5) #linie orizontala la 0
plt.axvline(0, color='black', linewidth=0.5) # linie verticala la 0
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.xlabel("Factor 1")
plt.ylabel("Factor 2")
plt.title("Cercul Corelatiilor fara rotatie")
plt.grid()
plt.show()

# cu rotatie
x1 = coreletii_factoriale_cu[:, 0]
y = coreletii_factoriale_cu[:, 1]
fig,ax = plt.subplots(figsize=(6,6))

circle = plt.Circle((0,0),1, color='b', fill=False, linestyle='dashed')
ax.add_patch(circle)
plt.scatter(x1,y)

for i, label in enumerate(x.columns):
    plt.text(x1[i], y[i], label, fontsize=9, ha="right", va="bottom")

plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.xlabel("Factor 1")
plt.ylabel("Factor 2")
plt.title("Cercul Corelatiilor cu rotatie")
plt.grid()
plt.show()

# Calcul comunalitati si varianta specifica
comunalitati = fa_cu_rotatie.get_communalities()
varianta_specifica = 1-comunalitati
print("Comunalitati cu rotatie: ", comunalitati)
print("Varianta specifica: ", varianta_specifica)

# Trasare corelograma comunalitati si varianta specifica
plt.figure(figsize=(10,5))
sb.barplot(x=x.columns, y=comunalitati)
plt.title("Corelograma comunitati")
plt.xticks(rotation=90)
plt.ylabel("Variabile")
plt.show()

# Calcul scoruri (cu/fara rotatie)
# fara rotatie
scoruri_fara = fa_fara_rotatie.transform(x)
scoruri_fara_df = pd.DataFrame(scoruri_fara, columns=[f"Scor{i+1}" for i in range(0,scoruri_fara.shape[1])])
print("Scoruri fara rotatie", scoruri_fara_df)

# cu rotatie
scoruri_cu = fa_cu_rotatie.transform(x)
scoruri_cu_df = pd.DataFrame(scoruri_cu, columns=[f"Scor{i+1}" for i in range(0, scoruri_cu.shape[1])])
print("Scoruri cu rotatie", scoruri_cu_df)

# Trasare plot scoruri
# fara rotatie
plt.figure(figsize=(10,8))
plt.scatter(scoruri_fara[:, 0], scoruri_fara[:, 1], c="Blue", alpha=0.8)
plt.title("Plot Scoruri Factoriale Fara Rotatie (Factor 1 vs Factor 2")
plt.xlabel("Factor 1")
plt.ylabel("Factor 2")
plt.show()

# cu rotatie
plt.figure(figsize=(10,8))
plt.scatter(scoruri_cu[:,0], scoruri_cu[:, 1], c="Red", alpha=0.8)
plt.title("Plot Scoruri Factoriale Cu Rotatie (Factor 1 vs Factor 2)")
plt.xlabel("Factor 1")
plt.ylabel("Factor 2")
plt.show()


# ------------------------------------------------------------------------------------------------------------------------------------------------
# => Analiza Discriminata (liniara) (LDA)

# importuri
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

# citire date
date = pd.read_csv("discriminant_analysis_data.csv")
x = date.drop(columns=["Target"])
y = date["Target"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)

#ad
lda = LinearDiscriminantAnalysis()
lda.fit(x_train, y_train)

# Calcul scoruri discriminante model liniar
scoruri = lda.transform(x_test)
print("Scoruri discriminante\n", scoruri)

# Trasare plot instante in axe discriminante
plt.figure(figsize=(10,8))
plt.scatter(scoruri[:, 0], [0]*len(scoruri), c=y_test, cmap="viridis", edgecolors='k', s=50)
plt.title("Instante in axe discriminante")
plt.xlabel("Axa discriminanta")
plt.yticks([])
plt.colorbar(label="Clasa")
plt.show()

# Trasare plot distributii in axele discriminante
lda_df = pd.DataFrame({"Scoruri discriminante": scoruri[:,0], "Clasa":y_test})
plt.figure(figsize=(10,8))
sb.histplot(lda_df, x="Scoruri discriminante", hue="Clasa", kde=True, bins=30, palette="viridis", alpha=0.6)
plt.title("Distributia scorurilor discriminante in axa LDA")
plt.xlabel("Scoruri Discriminante (Axa 1)")
plt.ylabel("Frecventa")
plt.show()

# Predictia in setul de testare model liniar
y_pred = lda.predict(x_test)
df_pred = pd.DataFrame({"Real": y_test.values, "Predictie":y_pred})
print("Predictia in setul de testare model liniar:\n", df_pred)

# Evaluare model liniar pe setul de testare (matricea de confuzie + indicatori de acuretete)
#matrice confuzie
matrice_confuzie = confusion_matrix(y_test, y_pred)
print("Matricea de confuzie:\n", matrice_confuzie)

#acuratete globala
acuratete_globala = accuracy_score(y_test, y_pred)
print("Acuratetea globala: ", acuratete_globala)

#acuratete medie
acuratete_per_clasa = matrice_confuzie.diagonal() / matrice_confuzie.sum(axis=1)
acuratete_medie = np.mean(acuratete_per_clasa)
print("Acuratete medie: ", acuratete_medie)

# Predictia in setul de aplicare model liniar
y_pred_aplicare = lda.predict(x_train)
df_pred_aplicare = pd.DataFrame({"Real": y_train.values, "Predictie": y_pred_aplicare})
print("Predictia in setul de aplicare model liniar:\n", df_pred_aplicare)

# Predictia in setul de testare model bayesian
#model bayesian
model_b = GaussianNB()
model_b.fit(x_train, y_train)
#predictia
predictie_b_test = model_b.predict(x_test)
print("Predictia in setul de testare model bayesian",predictie_b_test)

# Evaluare model bayesian (matricea de confuzie + indicatori de acuretete)
# Matricea de confuzie
matrice_confuzie_b = confusion_matrix(y_test, predictie_b_test)
print("Matricea de confuzie pentru modelul bayesian:\n", matrice_confuzie_b)

# Acuratete globala
acuratete_globala_b = accuracy_score(y_test, predictie_b_test)
print("Acuratetea globala pentru modelul bayesian: ", acuratete_globala_b)

# Acuratete medie
acuratete_per_clasa_b = matrice_confuzie_b.diagonal() / matrice_confuzie_b.sum(axis=1)
acuratete_medie_b = np.mean(acuratete_per_clasa_b)
print("Acuratete medie pentru modelul bayesian: ", acuratete_medie_b)

# Predictia in setul de aplicare model bayesian
predictie_b_aplicare = model_b.predict(x_train)
print("Predictie Bayes", predictie_b_aplicare)


# -------------------------------------------------------------------------------------------------------------------------------------
# => Analiza canonica (CCA)

# importuri necesare
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.cross_decomposition import CCA
from scipy.stats import chi2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

date = pd.read_csv('date_canonice.csv')
x = date.iloc[:, :3].values
y = date.iloc[:, 3:].values

cca = CCA(n_components=2)
cca.fit(x, y)

def test_bartlett(r2,n,p,q,m):
    x = 1-r2
    df=[(p-k+1)*(q-k+1) for k in range(1, m+1)]
    l = np.flip(np.cumprod(np.flip(x)))
    chi2_ = (-n+1+(p+q+1)/2)*np.log(1)
    return 1-chi2.cdf(chi2_, df)

# Calcul scoruri canonice(variabile canonice)
x_scoruri, y_scoruri = cca.transform(x, y)

# Calcul corelatii canonice
corelatii_canonice = np.corrcoef(x_scoruri.T, y_scoruri.T).diagonal(offset=x_scoruri.shape[1])
print("Corelatii canonice:\n", corelatii_canonice)

# Determinare relevanta radacine canonice (Test Bartllet)
r2 = corelatii_canonice*corelatii_canonice
n = len(date)
p = x.shape[1]
q = y.shape[1]
m = min(p,q,7)
p_values = test_bartlett(r2,n,p,q,m)
etichete_radacini = ["root" + str(i+1) for i in range(m)]
tabel_semnificatie = pd.DataFrame(
    data={
        "R":np.round(corelatii_canonice,5),
        "R2": np.round(r2,5),
        "P_Values": np.round(p_values, 5)
    }, index = etichete_radacini
)
tabel_semnificatie.index.name = "Radacina"
print(tabel_semnificatie)

nr_radacini = np.sum(p_values > 0.01)
nr_radacini = nr_radacini + 1 if nr_radacini == 1 else nr_radacini
print("Numarul de radacini semnificative este: ", nr_radacini)

# Calcul corelatii variabile observate - variabile canonice
corelatii_x_xscoruri = np.corrcoef(x.T, x_scoruri.T)[:x.shape[1], x.shape[1]:]
corelatii_y_yscoruri = np.corrcoef(y.T, y_scoruri.T)[:y.shape[1], y.shape[1]:]
print("Corelații X - X_c:\n", corelatii_x_xscoruri)
print("Corelații Y - Y_c:\n", corelatii_y_yscoruri)


# Trasare corelograma corelatii variabile observate - variabile canonice
df_corelatii_x = pd.DataFrame(corelatii_x_xscoruri,
                              index=[f"X{i+1}" for i in range(x.shape[1])],
                              columns=[f"X_Scor{i+1}" for i in range(x_scoruri.shape[1])])
df_corelatii_y = pd.DataFrame(corelatii_y_yscoruri,
                              index=[f"Y{i+1}" for i in range(y.shape[1])],
                              columns=[f"Y_Scor{i+1}" for i in range(y_scoruri.shape[1])])
plt.figure(figsize=(10, 8))
sb.heatmap(df_corelatii_x, annot=True, cmap="coolwarm", center=0)
plt.title('Corelograma variabile X - X_Scoruri')
plt.show()

plt.figure(figsize=(10, 8))
sb.heatmap(df_corelatii_y, annot=True, cmap="coolwarm", center=0)
plt.title('Corelograma variabile Y - Y_Scoruri')
plt.show()

# Trasare plot corelatii variabile observate - variabile canonice (cercul corelatiilor)
plt.figure(figsize=(8, 8))
plt.axhline(0, color='grey', lw=1)
plt.axvline(0, color='grey', lw=1)

circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='dashed')
plt.gca().add_patch(circle)

plt.scatter(df_corelatii_x.iloc[:, 0], df_corelatii_x.iloc[:, 1], label='X variables', color='blue')
plt.scatter(df_corelatii_y.iloc[:, 0], df_corelatii_y.iloc[:, 1], label='Y variables', color='red')
for i, label in enumerate(df_corelatii_x.index):
    plt.text(df_corelatii_x.iloc[i, 0], df_corelatii_x.iloc[i, 1], label, color='blue')
for i, label in enumerate(df_corelatii_y.index):
    plt.text(df_corelatii_y.iloc[i, 0], df_corelatii_y.iloc[i, 1], label, color='red')
plt.xlabel("Componenta canonică 1")
plt.ylabel("Componenta canonică 2")
plt.title("Cercul corelațiilor")
plt.legend()
plt.grid()
plt.show()

# Trasare plot instante in spatiile celor doua varianile (Biplot)
# Standardizare
scaler_x = StandardScaler()
scaler_y = StandardScaler()

x_std = scaler_x.fit_transform(x)
y_std = scaler_y.fit_transform(y)

# Aplicăm PCA pentru a reduce la 2 dimensiuni
pca_x = PCA(n_components=2)
pca_y = PCA(n_components=2)

x_pca = pca_x.fit_transform(x_std)
y_pca = pca_y.fit_transform(y_std)

# Trasare plot instante în spațiile variabilelor originale X și Y
plt.figure(figsize=(8, 8))
plt.scatter(x_pca[:, 0], x_pca[:, 1], label='Instanțe în spațiul X', color='blue')
plt.scatter(y_pca[:, 0], y_pca[:, 1], label='Instanțe în spațiul Y', color='red')

plt.xlabel("Componenta 1")
plt.ylabel("Componenta 2")
plt.title("Instanțe în spațiile variabilelor X și Y")
plt.legend()
plt.grid()
plt.show()

# Calcul varianta explicata si redundanta informationala
varianta_explicata = corelatii_canonice ** 2

var_explicata_x = np.var(x_scoruri, axis=0) / np.sum(np.var(x, axis=0))
var_explicata_y = np.var(y_scoruri, axis=0) / np.sum(np.var(y, axis=0))

redundanta_x = np.sum(varianta_explicata * var_explicata_x)
redundanta_y = np.sum(varianta_explicata * var_explicata_y)

print(f"Varianță explicată: {varianta_explicata}")
print(f"Redundanță informațională X: {redundanta_x:.4f}")
print(f"Redundanță informațională Y: {redundanta_y:.4f}")


# ------------------------------------------------------------------------------------------------------------
# => Analiza Cluster

# importuri
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

#citire date
date = pd.read_csv("date_clusterizare.csv")

# Calcul ierarhie (matricea ierarhie)
z = linkage(date, method="ward")
print("Matrice ierarhie:\n", z)

#si o dendrograma asa frumos
plt.figure(figsize=(10,7))
dendrogram(z)
plt.title("Dendrograma - Calcul Ierarhie")
plt.xlabel("Instante")
plt.ylabel("Distanta")
plt.show()

# Calcul partitie optimala (repartizarea instantelor in clusteri) prin metoda Elbow pe baza diferentelor dintre distantele de agregare (NU ELBOW KMEANS)
distanta = z[:,2]
diferente = np.diff(distanta,2)
punct_elb = np.argmax(diferente)+1
print("Numarul optim de clustere este: ", punct_elb)

#si un grafic ca asa e frumos
plt.figure(figsize=(8,6))
plt.plot(range(1, len(diferente)+1), diferente, marker='o')
plt.title("Diferentele dintre distantele de agregare")
plt.xlabel("Nr de pasi de agregare")
plt.ylabel("Diferenta distantei de agregare")
plt.show()

# Calcul partitie oarecare (cu un numar prestabilit de clusteri - numarul de clusteri se initializeaza prin cod sau se citeste de la tastatura/interfata grafica)
nr_clusteri = 5
clusteri_k = fcluster(z, t=nr_clusteri, criterion="maxclust")
print("Calcul pentru o partitie oarecare: ", clusteri_k)

# Calcul indecsi Silhouette la nivel de partitie si de instante
silhouette_per_instanta = silhouette_samples(date, clusteri_k)
silhouette_avg = silhouette_score(date, clusteri_k)
print(f"Indicele Silhouette pentru intreaga partitie: {silhouette_avg:.3f}")
print(f"Instante Silhouette pentru instante: {silhouette_per_instanta}")

# Trasare plot dendograma cu evidentierea partiei prin culoare (optimala si partitie-k)
# partitie optimala
plt.figure(figsize=(10,7))
dendrogram(z, color_threshold=distanta[punct_elb-1])
plt.title("Dendrograma - Calcul Ierarhie cu evidentierea partitie optimala")
plt.xlabel("Instante")
plt.ylabel("Distanta")
plt.show()

# Pentru partitia-k
plt.figure(figsize=(10,7))
dendrogram(z, color_threshold=distanta[nr_clusteri-1], above_threshold_color='b')
plt.title(f"Dendrograma - Calcul Ierarhie cu evidentierea partitie-{nr_clusteri}")
plt.xlabel("Instante")
plt.ylabel("Distanta")
plt.show()

# Trasare plot Silhouette partitie (optimala si partitie-k)
# Pentru partitia optimala
plt.figure(figsize=(10,7))
sb.barplot(x=silhouette_per_instanta, hue=range(len(silhouette_per_instanta)), palette='viridis')
plt.title("Silhouette pentru fiecare instanță - Partitia Optima")
plt.xlabel("Indicele Silhouette")
plt.ylabel("Instanțe")
plt.show()

# Pentru partitia-k
silhouette_per_instanta_k = silhouette_samples(date, clusteri_k)
plt.figure(figsize=(10,7))
sb.barplot(x=silhouette_per_instanta_k, hue=range(len(silhouette_per_instanta_k)), palette='viridis')
plt.title(f"Silhouette pentru fiecare instanță - Partitia-{nr_clusteri}")
plt.xlabel("Indicele Silhouette")
plt.ylabel("Instanțe")
plt.show()

# Trasare histograme clusteri pentru fiecare variabila observata (optimala si partitie-k)
date['Cluster'] = clusteri_k
# Histograme pentru variabila observata in partiția optimă
# Calcul partitie optimala
clusteri_optimal = fcluster(z, t=punct_elb, criterion="maxclust")

# Crearea histogramei pentru fiecare cluster
fig = plt.figure(figsize=(15, 9))
fig.suptitle("Histograme pentru variabila Variabila_1 - Partitia Optima")

# Identificarea valorilor unice ale clusterelor
clusteri = list(set(clusteri_optimal))
dim = len(clusteri)

# Crearea subgraficelor pentru fiecare cluster
axs = fig.subplots(1, dim, sharey=True)

# Iterăm prin fiecare cluster
for i in range(dim):
    ax = axs[i]
    ax.set_xlabel(f"Cluster {clusteri[i]}")

    # Selectăm valorile corespunzătoare clusterului curent
    cluster_mask = (clusteri_optimal == clusteri[i])
    cluster_values = date['Variabila_1'][cluster_mask]
    cluster_labels = np.array(date.index[cluster_mask])  # Folosim indecșii ca etichete

    # Crearea histogramei pentru clusterul curent
    counts, bins, bars = ax.hist(cluster_values, bins=10, rwidth=0.9, range=(min(date['Variabila_1']), max(date['Variabila_1'])), color='blue', alpha=0.7)

    # Adăugarea etichetelor la fiecare bară
    for bar, value, label in zip(bars, cluster_values, cluster_labels):
        ax.text(
            bar.get_x() + bar.get_width() / 2,  # Centrul barei
            bar.get_height(),  # Înălțimea barei
            label,  # Eticheta corespunzătoare
            ha='center', va='bottom', fontsize=8, rotation=45
        )

# Ajustăm layout-ul pentru a evita suprapunerea
plt.tight_layout()
plt.show()


# Histograme pentru variabila observata in partiția-k
# Crearea histogramei pentru fiecare cluster
fig = plt.figure(figsize=(9, 9))
fig.suptitle("Histograme pentru variabila Variabila_1")

# Identificarea valorilor unice ale clusterelor
clusteri = list(set(clusteri_k))
dim = len(clusteri)

# Crearea subgraficelor pentru fiecare cluster
axs = fig.subplots(1, dim, sharey=True)

# Iterăm prin fiecare cluster
for i in range(dim):
    ax = axs[i]
    ax.set_xlabel(f"Cluster {clusteri[i]}")

    # Selectăm valorile corespunzătoare clusterului curent
    cluster_mask = (clusteri_k == clusteri[i])
    cluster_values = date['Variabila_1'][cluster_mask]
    cluster_labels = np.array(date.index[cluster_mask])  # Folosim indecșii ca etichete (sau alte etichete relevante)

    # Crearea histogramei pentru clusterul curent
    counts, bins, bars = ax.hist(cluster_values, bins=10, rwidth=0.9, range=(min(date['Variabila_1']), max(date['Variabila_1'])), color='blue', alpha=0.7)

    # Adăugarea etichetelor la fiecare bară
    for bar, value, label in zip(bars, cluster_values, cluster_labels):
        ax.text(
            bar.get_x() + bar.get_width() / 2,  # Centrul barei
            bar.get_height(),  # Înălțimea barei
            label,  # Eticheta corespunzătoare
            ha='center', va='bottom', fontsize=8, rotation=45
        )

# Ajustăm layout-ul pentru a evita suprapunerea
plt.tight_layout()
plt.show()


# Trasare plt partitie in axe principale (optimala si partitie-k)
# Dimensiune redusa pentru partiția optimă
pca = PCA(n_components=2)
date_pca = pca.fit_transform(date)
plt.figure(figsize=(10,7))
plt.scatter(date_pca[:, 0], date_pca[:, 1], c=clusteri_k, cmap='viridis')
plt.title("Partiție în Axe Principale - Partitia Optima")
plt.xlabel("Componenta 1")
plt.ylabel("Componenta 2")
plt.show()

# Dimensiune redusa pentru partiția-k
plt.figure(figsize=(10,7))
plt.scatter(date_pca[:, 0], date_pca[:, 1], c=clusteri_k, cmap='viridis')
plt.title(f"Partiție în Axe Principale - Partitia-{nr_clusteri}")
plt.xlabel("Componenta 1")
plt.ylabel("Componenta 2")
plt.show()

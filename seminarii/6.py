import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.stats import f
from functii import calcul_metrici, salvare_ndarray
from grafice import f_distributii, f_scatter, show

# --- 1. Încărcarea și Pregătirea Datelor ---
# Folosim un set de date pentru diagnosticarea herniei
t = pd.read_csv("data_in/Hernia/hernia.csv", index_col=0)
variabile = list(t)
tinta = variabile[-1]  # Ultima coloană este variabila dependenta (clasa)
predictori = variabile[:-1] # Restul sunt variabilele independente

# Splitarea setului de date în Invățare (70%) și Testare (30%)
t_train, t_test, y_train, y_test = train_test_split(
    t[predictori], t[tinta], test_size=0.3, random_state=42
)

# --- 2. Evaluarea Puterii de Discriminare a Predictorilor ---
# Verificăm care variabile separă cel mai bine clasele (analiză tip ANOVA).
model_lda = LinearDiscriminantAnalysis()
model_lda.fit(t_train, y_train)

# Calculul statisticilor F pentru fiecare predictor
x = t_train[predictori].values
x_medie_generala = np.mean(x, axis=0)
g = model_lda.means_ # Mediile pe clase
n = len(t_train)
ponderi = model_lda.priors_ # Proporțiile claselor
q = len(ponderi)
dg = np.diag(ponderi) * n

# Matricele de împrăștiere (Sum of Squares)
sst = (x - x_medie_generala).T @ (x - x_medie_generala) # Totală
ssb = (g - x_medie_generala).T @ dg @ (g - x_medie_generala) # Inter-clase (Between)
ssw = sst - ssb # Intra-clase (Within)

# Statistica F și p-value pentru determinarea variabilelor semnificative
f_predictori = (np.diag(ssb) / (q - 1)) / (np.diag(ssw) / (n - q))
pvalues = 1 - f.cdf(f_predictori, q - 1, n - q)

df_predictori = pd.DataFrame({
    "Putere discriminare (F)": f_predictori,
    "P-Value": pvalues
}, index=predictori)
df_predictori.to_csv("data_out/Predictori_LDA.csv")

# Vizualizarea distribuțiilor pentru fiecare predictor pe clase
clase = model_lda.classes_
for predictor in predictori:
    f_distributii(t_train, predictor, tinta, clase)

# --- 3. Testarea și Validarea Modelului ---
predictie_test = model_lda.predict(t_test) # Aplicăm modelul pe datele de test

# Calculul metricilor de acuratețe (Matrice de confuzie, Kappa, Acuratețe medie)
t_cm, df_acuratete = calcul_metrici(y_test, predictie_test, clase)
t_cm.to_csv("data_out/CM_lda.csv")
df_acuratete.to_csv("data_out/Acuratete_lda.csv")

# --- 4. Analiza Scorurilor Discriminante (Z) ---
# Transformăm datele în spațiul axelor discriminante (Z1, Z2...).
z = model_lda.transform(t_train)
etichete_z = ["Z" + str(i + 1) for i in range(q - 1)]
t_z = salvare_ndarray(z, t_train.index, etichete_z, "data_out/z.csv")

# Calculul centrelor de greutate în noul spațiu
t_gz = t_z.groupby(by=y_train).mean()
f_scatter(t_z, t_gz, y_train, clase) # Plotarea instanțelor în axe discriminante

show()
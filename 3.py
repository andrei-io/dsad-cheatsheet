import numpy as np
import pandas as pd


def calcul_entropie_distributie(dataframe_date: pd.DataFrame):
    """
    Calculeaza Entropia Shannon (un index al diversitatii sau
    al distributiei) pentru fiecare coloana (categorie de votanti).

    O entropie mare inseamna ca votantii dintr-o categorie (ex: Barbati_18-24)
    sunt distribuiti uniform in toate localitatile (liniile) analizate.
    O entropie mica inseamna ca sunt concentrati in cateva localitati.

    (Logica originala a fost pastrata)

    Argumente:
        dataframe_date (pd.DataFrame): Un DataFrame unde liniile sunt
                                     localitati si coloanele sunt
                                     categorii de votanti (fostul 't').

    Returneaza:
        pd.Series: O serie cu valoarea entropiei pentru fiecare coloana.
    """
    # Extrage valorile intr-o matrice NumPy (fostul 'x')
    matrice_valori = dataframe_date.values

    # Calculeaza totalul pe fiecare coloana (categorie) (fostul 'tx')
    total_coloane = np.sum(matrice_valori, axis=0)
    # Previne impartirea la zero
    total_coloane[total_coloane == 0] = 1

    # Calculeaza matricea de proportii (fostul 'p')
    # p[i, j] = proportia de votanti din localitatea 'i'
    #           din totalul categoriei 'j'
    proportii = matrice_valori / total_coloane

    # Gestioneaza proportiile egale cu zero (logica originala)
    # Limita lui p*log2(p) cand p->0 este 0.
    # Setand p=1, termenul p*log2(p) devine 1*log2(1) = 0.
    proportii[proportii == 0] = 1

    # Calculeaza Entropia Shannon (fostul 'h')
    # E[j] = - Suma_i ( p[i,j] * log2(p[i,j]) )
    entropie = -np.sum(proportii * np.log2(proportii), axis=0)

    # Returneaza rezultatul ca o Serie Pandas
    return pd.Series(entropie, dataframe_date.columns)


# Seteaza optiunea Pandas sa afiseze toate coloanele unui DataFrame
pd.set_option("display.max_columns", None)

# --- 1. Incarcarea si Pregatirea Datelor ---

# Incarca setul de date principal
df_prezenta_vot = pd.read_csv("data_in/prezenta_vot.csv", index_col=0)

# Obtine lista tuturor coloanelor
lista_coloane_totale = list(df_prezenta_vot)
# Selecteaza doar coloanele care contin date despre vot (de la a 4-a coloana)
coloane_vot = lista_coloane_totale[3:]
# Selecteaza doar coloanele demografice (de la "Barbati_18-24" incolo)
index_start_categorii = coloane_vot.index("Barbati_18-24")
coloane_categorii_demografice = coloane_vot[index_start_categorii:]

# --- Cerinta 1 (implicita): Calcul Procent Participare ---

# Calculeaza procentul de participare folosind formula data
# (LT = Total votanti pe liste)
procentaj = (
    df_prezenta_vot["LT"]
    * 100
    / (df_prezenta_vot["Votanti_LP"] + df_prezenta_vot["Votanti_LS"])
)

# Creeaza un nou DataFrame pentru acest procentaj
df_procent_participare = pd.DataFrame(procentaj, columns=["ProcentParticipare"])
# Insereaza coloana "Localitate" la inceput
df_procent_participare.insert(0, "Localitate", df_prezenta_vot["Localitate"])

# Salveaza localitatile cu prezenta de peste 50%
df_procent_participare[df_procent_participare["ProcentParticipare"] > 50].to_csv(
    "data_out/Prezenta50.csv"
)

# --- Cerinta 2 (implicita): Sortare dupa Prezenta ---

# Sorteaza DataFrame-ul cu procentajul participarii in ordine descrescatoare
df_prezenta_sortata = df_procent_participare.sort_values(
    by="ProcentParticipare", ascending=False
)
# Salveaza rezultatul sortat
df_prezenta_sortata.to_csv("data_out/PrezentaSort.csv")

# --- Cerinta 3: Agregare la Nivel de Regiune ---

# Incarca fisierul de mapare Judet -> Regiune
df_coduri_judet = pd.read_csv("data_in/Coduri_Judete.csv", index_col=0)

# Combina (merge) datele despre prezenta cu cele despre regiuni
# Se face legatura intre coloana "Judet" si indexul din 'df_coduri_judet'
df_prezenta_cu_regiuni = df_prezenta_vot.merge(
    df_coduri_judet, left_on="Judet", right_index=True
)

# Selecteaza coloanele de vot si coloana "Regiune"
# Grupeaza dupa "Regiune" si insumeaza valorile
df_agregat_regiune = (
    df_prezenta_cu_regiuni[coloane_vot + ["Regiune"]].groupby(by="Regiune").sum()
)
# Salveaza totalurile pe regiuni
df_agregat_regiune.to_csv("data_out/Regiuni.csv")

# --- Cerinta 4: Categoria de Varsta Dominanta pe Localitate ---

# Aplica o functie pe fiecare linie (axis=1) a datelor demografice
# Functia lambda gaseste numele coloanei (x.index) care are valoarea maxima (x.argmax())
categorie_dominanta = df_prezenta_vot[coloane_categorii_demografice].apply(
    func=lambda linie: linie.index[linie.argmax()], axis=1
)

# Creeaza un DataFrame pentru acest rezultat
df_categorie_dominanta = pd.DataFrame(categorie_dominanta, columns=["Categorie"])
# Insereaza coloana "Localitate"
df_categorie_dominanta.insert(0, "Localitate", df_prezenta_vot["Localitate"])
# Salveaza rezultatul
df_categorie_dominanta.to_csv("data_out/Varsta.csv")

# --- Cerinta 5: Filtrare Localitati dupa Categoria Dominanta ---

# Defineste categoria de varsta cautata
categorie_tinta = "Barbati_45-64"

# Filtreaza DataFrame-ul de la cerinta 4
# Pastreaza doar liniile unde "Categorie" este egala cu 'categorie_tinta'
df_localitati_cu_categorie_tinta = df_categorie_dominanta[
    df_categorie_dominanta["Categorie"] == categorie_tinta
]
# Salveaza lista filtrata de localitati
df_localitati_cu_categorie_tinta.to_csv("data_out/" + categorie_tinta + ".csv")

# --- Cerinta 6: Calcul Entropie (Disparitate) pe Judet ---

# Selecteaza coloanele demografice si coloana "Judet"
# Grupeaza datele pe "Judet"
# Aplica functia 'calcul_entropie_distributie' pentru fiecare grup (judet)
# Functia va calcula entropia distributiei geografice (intre localitatile
# din acel judet) pentru fiecare categorie demografica.
df_entropie_pe_judet = (
    df_prezenta_vot[coloane_categorii_demografice + ["Judet"]]
    .groupby(by="Judet")
    .apply(func=calcul_entropie_distributie, include_groups=False)
)
# print(df_entropie_pe_judet)
# Salveaza matricea rezultat (Linii=Judete, Coloane=Categorii, Valori=Entropie)
df_entropie_pe_judet.to_csv("data_out/Disparitate_vot.csv")

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

"""

Import des data

"""

base_train_data = pd.read_csv("kanji_train_data.csv")
target_data = pd.read_csv("kanji_train_target.csv")

"""

Ajout des classes pour le sample / drop (j'aurais pu faire avec un indice sample de la sorte : 

index_sample = int(train_data.shape[0]*0.7)

X_train = train_data[:index_sample]
Y_train = train_target[:index_sample]

X_valid = train_data[index_sample:]
Y_valid = train_target[index_sample:]

=> je l'utilise dans les prochains

)

"""

train_data = base_train_data.copy()
train_data["classe"] = target_data.iloc[:, 0]

nb_classes = len(train_data["classe"].unique()) #20

train_data = train_data.fillna(value = train_data.mean())

kanji_train_data = train_data.sample(frac=0.8, random_state=90)
kanji_valid_data = train_data.drop(kanji_train_data.index)

X_kanji_train = kanji_train_data.drop(columns=['classe'])
Y_kanji_train = kanji_train_data['classe']

X_kanji_valid = kanji_valid_data.drop(columns=['classe'])
Y_kanji_valid = kanji_valid_data['classe']


"""

Calcul de la distance

"""

def neighbors(X_train, y_label, x_test, k):
    
    list_distances = []
    list_distances = np.sum((X_train.values - np.expand_dims(x_test,axis=0))**2,1)

    df = pd.DataFrame()

    df["classe"] = y_label
    df["distance"] = list_distances

    df = df.sort_values(by="distance")

    return df.iloc[:k, :]


"""

Prédiction :

On regarde la classe des plus proches voisin et on stock ça dans un tableau indicé (représentant la repartition des classes)
On recupère ensuite la classe la plus représentée

"""


def prediction(neighbors):
    classes_possibles = np.zeros(nb_classes)
    for voisin in neighbors["classe"]:
        classes_possibles[voisin] += 1

    classe_choisie = np.where(classes_possibles == max(classes_possibles))[0][0]
    return classe_choisie


"""

Eval des k-voisins

"""


def evaluation(X_train, Y_train, X_valid, Y_valid, k, detail=True):
    bon_guess = 0
    mauvais_guess = 0

    total = 0

    for i in tqdm(range(X_valid.shape[0])):
        nearest_neighbors = neighbors(X_train, Y_train, X_valid.iloc[i], k)

        if((prediction(nearest_neighbors) == Y_valid.iloc[i])):
            bon_guess += 1

        else:
            mauvais_guess += 1

        total += 1

    precision = bon_guess / total

    if detail:
        print("Bon guess : ", bon_guess)
        print("Mauvais guess : ", mauvais_guess)
        print("Précision : ", precision)
    
    return precision


"""

Test des K-voisins

"""

"""

list_precisions = []

for k in tqdm(range(1, 19, 1)): #mon graphique a un pas de 2 mais il n'est pas nécéssaire, on peut faire un pas de 1
    list_precisions.append(evaluation(X_kanji_train, Y_kanji_train, X_kanji_valid, Y_kanji_valid, k, False))

print(list_precisions)

x=range(1,19,2)
y=list_precisions

plt.plot(x,y)
plt.xlabel("k")
plt.ylabel("accuracy")
plt.show()

"""

"""

Test du modèle :

le k le plus performant est le 1 ici. (voir graph)
0.95 de succès sur les données de test

"""

list_predictions = []
test_data = pd.read_csv("kanji_test_data.csv")
print(test_data)
for i in tqdm(range(test_data.shape[0])):
    nearest_neighbors = neighbors(X_kanji_train, Y_kanji_train, test_data.iloc[i], 1)
    list_predictions.append(prediction(nearest_neighbors))


pd.DataFrame(list_predictions).to_csv("kanji_test_predictions.csv", index=False)

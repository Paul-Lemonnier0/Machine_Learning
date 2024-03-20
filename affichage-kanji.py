import matplotlib.pyplot as plt
import pandas as pd

kanjis_data = pd.read_csv("kanji_train_data.csv") 

def print_kanji(kanji_data_array):
    if(len(kanji_data_array) == 4096):
        #reshape en matrice de 64x64
        kanji_matrice = kanji_data_array.reshape(64,64)
        #cmap a gray pour eviter le violet / jaune
        plt.imshow(kanji_matrice, cmap='gray')
        #pas d'axes afficher (ni valeur de pas)
        plt.axis('off')
        plt.show()

print_kanji(kanjis_data.iloc[0].values)

def affiche_100_kanjis():

    #on prend les 100 premiers
    kanjis_data_sub = kanjis_data.head(100)

    kanjis_reshaped = []

    for i in range(100):
        #on prend les reshape tous en une matrice de 64 par 64 pixels
        reshaped_kanji = kanjis_data_sub.iloc[i].values.reshape(64, 64)
        kanjis_reshaped.append(reshaped_kanji)

    taille_grille = 10

    #on créer des subplots pour les mettres sous forme de grille (10x10)
    fig, axes = plt.subplots(taille_grille, taille_grille, figsize=(10, 10))

    #ajout de subplot => imshow pour la conversion de matrice de pixels en image

    for i in range(taille_grille):
        for j in range(taille_grille):
            index = i * taille_grille + j
            if index < len(kanjis_reshaped):

                #cmap a gray pour eviter la couleur violette et jaune
                axes[i, j].imshow(kanjis_reshaped[index], cmap='gray')

                #on enleve la barre des axes (abscices, ordonnées)
                axes[i, j].axis('off')

    plt.show()

#affiche_100_kanjis()

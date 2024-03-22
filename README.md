# Détéction de Kanji

Implémentation d'algorithme pour la détéction de kanji écris à la main

## Description

Dans ce répertoire, plusieurs algorithme de Deep Learning sont implémentés.  
   
L'objectif principal est, grâce à un jeu de données (kanji_train_data.csv associé à kanji_target_data.csv), effectuer des prévisions sur l'appartenant d'un kanji manuscrit à une classe.  

## Algorithmes implémentés

K-plus proches voisins **(K-voisins.py)** (95% de réussite, voir le graphe **Kanji_data-NN.py** pour le k le plus performant (ici 1))  
Régression logistique multivariée **(Regression_log.py)** (70% de réussite)  
Réseau de neurones linéaire **(Reseau_neurones_lineaires.py)** (85% de réussite)  
Réseau neuronal convolutif **(CNN_torch.py)** (97,5% de réussite)  

Un kanji (ou plusieurs) peut-être afficher grâce à ce script : **affichage-kanji.py**   

## Utilisation

-Dézipper le dossier **Kanji_Data.zip**  
-Placer les 4 fichiers dans le même dossier que les scripts  
-Compiler le programme voulu (ils sont de base programmé pour être utiliser sur un GPU, voir la partie "device" et la changer en CPU si cela pose problème. Il faudra aussi enlever les .cpu())    

## Help

GPU introuvable :

```
Passer la variable device de cuda:0 à cpu
Enlever les appels .cpu()

/!\ l'éxécution du programme est beaucoup plus lente sur cpu

```

## Auteur

Paul Lemonnier     
paul.lemonnier49070@gmail.com  

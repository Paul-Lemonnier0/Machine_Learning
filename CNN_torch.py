import numpy as np
import torch as th
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from tqdm import tqdm
import pandas as pd

"""

Classe du model et primitive

"""

# fonction qui renvoie la prediction a partir de la sortie du modele (softmax(output(...)))
def prediction(sortie_modele):
    return th.argmax(sortie_modele, dim=1)

# taux de précision
def accuracy(pred, y):
    return np.count_nonzero(pred == y)/pred.shape[0]


class CNN(th.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        #1 en entrée (1 valeur a assimilée = pixel) 32 filtres 5x5
        self.conv1 = th.nn.Conv2d(1, 32, 5)

        #32 filtres en entrées, 64 en sorties, tjrs 5x5
        self.conv2 = th.nn.Conv2d(32, 64, 5)

        #choix de 512 neurones dans cette couche (TODO : check si ça impact la précision)
        self.layer1 = th.nn.Linear(64*13*13, 512)

        #512 entrées, 20 sorties (TODO : idem)
        self.layer2 = th.nn.Linear(512, 20)

    def forward(self, x):
        
        #Fonction d'activation choisie : laisse passer les valeurs positives et bloque les valeurs négatives
        x = th.nn.functional.relu(self.conv1(x))
        #Réduit la dimension de l'espace de calcul en récupérant la valeur max (dans un espace de ici 2x2 => arbitraire mais fonctionne bien)
        x = th.nn.functional.max_pool2d(x, 2)

        x = th.nn.functional.relu(self.conv2(x))
        x = th.nn.functional.max_pool2d(x, 2)

        #Reshape pour traitement layer
        x = x.view(-1, 64*13*13)

        x = th.nn.functional.relu(self.layer1(x))
        x = self.layer2(x)

        #proba d'appartenir à une classe (+ stable que softmax tout court)
        return th.nn.functional.log_softmax(x, dim=1)
    


#création d'une classe "convertisseuse" d'un csv vers un dataset :
class KanjiDataset(Dataset):
    def __init__(self, data_path, target_path):
        super().__init__()
        self.data = np.genfromtxt(data_path, delimiter=',')
        self.data = self.data.reshape(-1, 1, 64, 64).astype(np.float32)
        self.target = np.genfromtxt(target_path, delimiter=',').astype(np.int64)

    #OVERRIDE

    #ré-implementation de la fonction len() de la classe mère Dataset (utilisé implécitement dans les fonctions pré-faites)
        
    def __len__(self):
        return len(self.data)
    
    #idem pour getitem
    def __getitem__(self, index):
        data = self.data[index]
        target = self.target[index]
        return data, target

"""

Importation

"""    
chemin_train_data = "kanji_train_data.csv"
chemin_target_data = "kanji_train_data.csv"

dataset = KanjiDataset(chemin_train_data, chemin_target_data)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

"""

Model

"""

device = th.device("cuda:0") #ou cpu di CPU 

model = CNN()
model = model.to(device)

"""

Hyperparamètres

"""

"""

EPA : 

Meilleure valeur pour l'instant : 0.0001, permet d'obtenir 0.9997 sur le train et 0.978 sur le test

"""

eta = 0.0001

"""
EPOCH :

J'ai tester pour différentes valeurs : 

[20, 15, 10, 8, 6, 4, 2, 1]

20, 10, 6, 2, 1 sont aussi efficaces sur les données train avec le max : 0.9997
Par contre 20 se démarque sur les données de test avec 0.978 contre 0.974 pour 10

"""

num_epochs = 20

weight_decay = 0.001 #meilleure valeur aussi (pas fais de graph pour celle ci encore)

optimizer = optim.Adam(model.parameters(), lr=eta, weight_decay=weight_decay)
criterion = th.nn.CrossEntropyLoss()

"""

TRAINING

"""

X_test = np.genfromtxt("/content/kanji_test_data.csv", delimiter=',')

X_test = X_test.reshape(-1, 1, 64, 64)
X_test = th.tensor(X_test, dtype=th.float32).to(device)

#num_epochs_list = [20, 15, 10, 8, 6, 4, 2, 1]
num_epochs_list = [20] #meilleure valeur sur bcp de situation

list_eta = [10, 5, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001] # pas encore tout testé

list_accuracy = []

for num_ep in tqdm(num_epochs_list):
  # Entraînement du modèle
  for epoch in tqdm(range(num_ep)):

      model.train()
      #loss => pour checker le sur-entrainement
      running_loss = 0.0
      correct_predictions = 0
      total_predictions = 0

      for batch_idx, (data, target) in enumerate(train_loader):
          data = data.to(device)
          target = target.to(device)

            #remise à 0 du gradient
          optimizer.zero_grad()
          output = model(data)

          loss = criterion(output, target)
          loss.backward()
          optimizer.step()

          running_loss += loss.item()

          y_pred = prediction(output, 1)

          correct_predictions += (y_pred == target).sum().item()
          total_predictions += target.size(0)
          #ou utilisation de la méthode accuracy

      accuracy_train = correct_predictions / total_predictions
      #print("Epoch:", epoch+1, "Loss:", running_loss/len(train_loader), "Accuracy:", accuracy_train)

  print("epoch : ", epoch, " || accuracy : ", accuracy_train)

"""

TEST

"""
model.eval()
predictions = []
for data in X_test:
    data = data.to(device)
    output = model(data)
    y_pred = prediction(output, 1)
    predictions.extend(y_pred.cpu().numpy())

#print(predictions)
pd.DataFrame(predictions).to_csv("kanji_test_predictions.csv", index=False)

#J'ai juste un soucis ici, quand le fichier se créer il y a un 0 a la premiere ligne qui ne devrait pas être la.
#Pour l'instant je le fais a la main je verrais ça dans la semaine

#pd.DataFrame(predictions).to_csv("kanji_test_predictions.csv", index=False)
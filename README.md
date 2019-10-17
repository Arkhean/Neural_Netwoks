# Neural_Networks

Implémentations et tests de réseaux de neurones standards et aussi récursifs.

Le réseau de neurones standards est configurable (nombre de niveaux, dropout, ...)

Deux méthodes d'apprentissage sont possibles :
* par descente graduelle
* par algoritme génétique

Possibilité de choisir les fonctions d'activation (sigmoide, tangente, relu).

# Tests

Le réseau standard est testé sur IRIS ainsi que sur MNIST.

Le réseau récursif est testé sur l'addition binaire (retenue...)

# Librairie dynamique

On peut compiler le projet sous forme de librairie : 

make networks.so

On peut alors l'utiliser sur un test (sans oublier LD_LIBRARY_PATH) :

g++ test.cpp -L. -lnetworks

# Note

Le logiciel utilise une partie du code de Baptiste Wicht disponible ici : https://github.com/wichtounet/mnist pour la lecture des fichiers contenant la base de données MNIST.

GLOBALEMENT : 
mettre les csv dans data : RES-6-9 et RES-6-9-labels



POUR GIT 
Cloner le repo depuis le dossier cible 

git clone "https url..."

créer une branche si besoin 

Ouvrir visual studio 
File- Add folder to workspace (Dossier git)

Créer environnement virtuel 
python -m venv .venv 
Activer l'environnement virtuel 
.\.venv\Scripts\Activate.ps1 (sur windows)
Installer les paquets dans l'environnement virtuel 
pip3 install -r requirements.txt

Avant de commit : 
git pull (Pour bien vérifier que nous sommes à jour)
Faire le commit:
Aller sur le logo branch, stage les changements et mettre un commentaire. 
Synchroniser le commit: 
git push. 

Si des paquets additionnels ont été ajoutés dans l'environnement : 
pip freeze > requirements.txt puis commit et push les changements. 


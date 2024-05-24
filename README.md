# Prédire le changement climatique

Ce dépôt contient tout le code nécessaire pour mettre en place un modèle d'intelligence artificielle de prédiction de tempoérature disponible via une API. Ce modèle est adapté pour prédire l'évolution de températures mensuelles moyennes et a été testé avec [ce jeu de données](https://www.kaggle.com/datasets/berkeleyearth/climate-change-earth-surface-temperature-data).

## Mise en place

### Déploiement de l'API

Le modèle utilisé est un modèle **SARIMA**. Ses hyperparamètres ayant déjà été définis en travaillant avec le jeu de données cité ci-dessus, il n'est pas nécessaire de les modifier. Voici la marche à suivre pour déployer le modèle via une API sur un serveur Linux :
- Pour commencer, clonez ce dépôt GitHub :
  ```git clone https://github.com/vinpap/predict_climate_change.git```
- Dans le répertoire où vous avez cloné le dépôt, installez toutes les dépendances du projet :
  ```pip install -r requirements.txt```
- Définissez un token de sécurité pour votre API. **Ce token doit uniquement être connu des personnes autorisées à utiliser l'API dans votre organisation** (plus d'informations dans la section "Notes concernant la sécurité) :
  ```export API_TOKEN="<Valeur choisie pour votre token de sécurité>"```
- Lancez l'API :
  ```fastapi run api.py```
Si vous souhaitez déployer l'API sur un service de cloud (Azure, AWS...) référez-vous à la documentation de votre fournisseur de cloud pour connaître la marche à suivre pour déployer une application web sur ses serveurs.

L'API est maintenant disponible sur vos serveurs. Cependant, le modèle d'intelligence artificielle n'a pas encore été initialisé. Le premier entraînement du modèle est réalisé par le **script de monitorage** dont la mise en place est expliquée dans la section suivante.

### Installation du système de monitorage des performances

L'API est fournie avec un système de monitorage des performances du modèle d'intelligence artificielle codé dans le script **monitoring.py**. Ce script destiné à être exécuté régulièrement teste les performances du modèle et alerte automatiquement les administrateurs de l'API dans votre organisation pour les alerter si les performances du modèle se sont dégradées. Dans cette section, nous allons voir comment mettre en place cette fonctionnalité.

> **_REMARQUE :_**  Bien que vous puissiez exécuter manuellement ce script lorsque vous voulez vérifier les performances de votre modèle, il est recommandé d'automatiser son exécution afin d'éviter les oublis et de garantir un bon suivi du modèle. Vous pouvez héberger le script de monitorage sur votre serveur local ou bien utiliser un service de cloud tel qu'Azure Functions.

Deux fichiers nous intéressent dans le cadre du monitorage : **monitoring.py** et **monitoring_cfg.yml**. L'essentiel de la configuration est réalisé dans le fichier monitoring_cfg.yml. Ouvrez donc ce fichier et suivez les étapes suivantes :
- **Spécifiez l'adresse de l'API**. Éditez la valeur "API_ENDPOINT" en indiquant l'URL où vous avez déployé l'API
- **Définissez la source des données**. Pour fonctionner, le script a besoin d'accéder aux températures mensuelles moyennes relevées les plus récentes. Afin de faciliter l'intégration du monitorage à vos systèmes de données existants, le script peut utiliser deux types de sources de données différents : un **fichier CSV** ou une **base de données PostgreSQL** :
  - Si vous souhaitez récupérer les données depuis un fichier CSV, modifiez la section "DATA" comme suit :
  ```
  DATA:
    LOAD_FROM_CSV: true
    CSV_PATH: <INDIQUEZ LE CHEMIN OU L'URL QUI POINTE VERS LE FICHIER CSV À UTILISER>
  ```
  Le fichier CSV devra contenir deux colonnes : une colonne **temperatures** qui stocke les températures mensuelles en degrés Celsius et une colonne **measurement_dates** qui stocke les dates de chaque relevé
  - Si vous préférez collecter les données depuis une base de données PostgreSQL, remplissez la secion "DATA" de cette manière :
  ```
  DATA:
    LOAD_FROM_CSV: false
    DB_NAME: <le nom de votre base de données>
    DB_PORT: <le port à utiliser>
    DB_URL: <l'adresse de l'hôte>
    DB_USER: <nom de l'utilisateur sur le serveur PostgreSQL>
  ```
  Les données devront être stockées dans une table appelée **monitoring** qui devra contenir une colonne **temperatures** pour les températures mensuelles moyennes en degrés Celsius et une colonne **measurement_dates** pour les dates correspondant à chaque température
- **Paramétrez le serveur SMTP** : un serveur SMTP est nécessaire pour envoyer automatiquement des alertes par e-mail aux administrateurs de l'API en cas de problème ou de dégradation des performances du modèle. Vous pouvez utiliser votre propre serveur SMTP que vous exécutez localement, ou utiliser le serveur SMTP d'un tiers comme celui de GMail par exemple :
```
EMAIL_SETTINGS:
  EMAIL_RECIPIENT: <adresse e-mail ou les alertes doivent être envoyées>
  SMTP_SERVER: <URL du serveur SMTP à utiliser, par exemple smtp.gmail.com pour le serveur GMail>
```
- Avant d'exécuter le script monitoring.py, vous devez encore définir plusieurs variables d'environnement. Exécutez ces commandes :
```
export API_TOKEN=<token de l'API>
export SMTP_LOGIN=<votre identifiant sur le serveru SMTP>
export SMTP_PASSWORD=<votre mot de passe sur le serveur SMTP>
export DB_PWD=<mot de passe sur le serveur PostgreSQL. Ignorez cette ligne si vous utilisez un fichier CSV à la place>
```
Ces valeurs ne sont pas stockées dans le fichier de configuration car elles doivent rester confidentielles.
- Enfin, exécutez la commande ```python monitoring.py --setup```. Cette commande entraînera une première version du modèle d'intelligence artificielle de l'API et déterminera automatiquement le niveau de performances que le modèle doit maintenir
Votre API ainsi que le système de monitorage du modèle d'intelligence artificielle sont maintenants opérationnels ! Vous serez automatiquement avertis par e-mail si les performances de votre modèle se sont dégradées ou si le script de monitorage a automatiquement réentraîné votre modèle. N'oubliez pas de lancer régulièrement le script de monitorage en lui fournissant des températures mensuelles récentes. Pour cela, exécutez la commande ```python monitoring.py```.

## Réentraîner le modèle

Il est possible que vous souahitez réentraîner vous-même le modèle d'intelligence artificielle de l'API, par exemple si vous souhaitez modifier le modèle pour prédire des températures propres à un lieu spécifique plutôt que des températures globales. La solution la plus simple et la plus propre dans ce cas consiste relancer la commande ```python monitoring.py --setup``` après avoir paramétré la source de données en suivant la procédure de la section précédente.

> **_ATTENTION :_**  Cette opération est irréversible. Réentraîner le modèle de l'API effacera le modèle précédemment utilisé. Réalisez plusieurs installations séparées de l'API si vous souhaitez conserver plusieurs modèles différents.

## Notes concernant la sécurité

Une attention particulière doit être portée au token utilisé pour sécuriser l'API. Ce token doit être envoyé avec chaque requête à l'API et sert à la protéger des accès non autorisés. Voici plusieurs bonnes pratiques à respecter pour que l'API reste sécurisée :
- Évitez de choisir un token trop court ou trop simple. Il est préférable de générer aléatoirement une longue suite de caractères
- Communiquez le token de l'API aux utilisateurs via des canaux sécurisés
- Changez régulièrement le token
- En plus des règles de sécurité concernant le token, mettez en place une liste d'adresses IP autorisées à accéder à l'API.

## Utilisation de l'API

La section suivante est destinée aux météorologues qui souhaiteraient utiliser l'API pour obtenir des prédictions de température :
- Obtenez le token de sécurité de l'API auprès des administrateurs de votre organisation. **Ce token doit rester secret**
- Envoyez une requête POST à l'adresse suivante : ```<adresse où est déployée l'API dans votre organisation>/predict```. Deux valeurs devront être jointes à cette requête :
  - 'date' : date pour laquelle vous souhaitez obtenir une prédiction, en format mm/aaaa
  - 'secret_token' : la valeur du token de sécurité
  
  En réponse, l'API vous enverra les températures mensuelles prédites pour chaque mois jusqu'à la date que vous avez envoyée. La réponse se compose de deux listes : une liste 'temperatures' qui contient les valeurs prédites, et une liste 'dates' qui contient les dates associées à chaque température.

Rapprochez-vous des administrateurs de l'API dans votre organisation si vous rencontrez des difficultés.



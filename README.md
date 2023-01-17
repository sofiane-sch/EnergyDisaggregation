# EnergyDisaggregation Package


## Installation
 1. First you need to install poetry, you will find all ressources on  [python-poetry](https://python-poetry.org/docs/)

 2. To install the energydisaggregation package, run the following command:
   ```poetry install```

This should install the package in a virtual environnement that you will be using during the project. If you need to add new library to the project please do it by using the following command :
```poetry add NewPackage```

## Folder structure

- `energydisaggregation/`: This is the root folder of your package, it contains the actual code of your package.
  - `energydisaggregation/data`: data treatement and data loading module
    - `energydisaggregation/data/config`: Contains information about the columns of interest for each data source
    - `energydisaggregation/data/dataloader`: data treatement and data loading class
- `Notebook/`: This is where you should place your jupyter notebooks

## Start
Please review the `Notebook/EDA.ipynb` notebook to understand how to manipulate data and start you exploratory data analysis.

## Data
We collected for you history of regional french power consumption and weather data : 
- `Data/consommation-quotidienne-brute-regionale.csv` contains power consumption history extracted from [RTE opendata source](https://odre.opendatasoft.com/explore/dataset/consommation-quotidienne-brute-regionale/export/?disjunctive.code_insee_region&disjunctive.region)
- `Data/donnees-synop-essentielles-omm.csv` contains weather information history extracted from [SYNOP opendata source](https://public.opendatasoft.com/explore/dataset/donnees-synop-essentielles-omm/table/?flg=fr&sort=date)

Please **download** them into the `Data/` forlder
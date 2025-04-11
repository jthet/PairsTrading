# test all strats

# use default pairs file
python backend/BaseBacktest.py --no-plot 
python backend/LogisticStrategy.py --no-plot 
python backend/XGBoostStrategy.py --no-plot 
python backend/ZScoreStrategy.py --no-plot 

# python backend/PairsFinder.py # this creates data/100_clusters_pairs.csv 

# python backend/BaseBacktest.py --no-plot --pairs-file data/100_clusters_pairs.csv
# python backend/LogisticStrategy.py --no-plot --pairs-file data/100_clusters_pairs.csv
# python backend/XGBoostStrategy.py --no-plot --pairs-file data/100_clusters_pairs.csv
# python backend/ZScoreStrategy.py --no-plot --pairs-file data/100_clusters_pairs.csv


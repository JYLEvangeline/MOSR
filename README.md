# MOSR

The dataset could be download in https://www.cs.cmu.edu/~enron/. Please put it in data/
Copyright: data/organization2.csv

# To run our code:
cd code
# generate the .json file we need
python check_data_distance_2.py  
# EnronA 
python model_new.py --md 10 -v 0 -lr 0.99
# EnronB
python model_new.py --md 10 -v 1 -lr 0.99


python create_arithmetic_dataset.py
head -n 9501 data.csv > arithmetic_train.csv


rm arithmetic_validation.csv
head -n 1 data.csv >> arithmetic_validation.csv
tail -n 500 data.csv >> arithmetic_validation.csv

rm arithmetic_test.csv
cp arithmetic_validation.csv arithmetic_test.csv



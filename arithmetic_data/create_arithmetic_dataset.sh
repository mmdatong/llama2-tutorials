
python create_arithmetic_dataset.py
head -n 9501 data.csv > arithmetric_train.csv


rm arithmetric_validation.csv
head -n 1 data.csv >> arithmetric_validation.csv
tail -n 500 data.csv >> arithmetric_validation.csv

rm arithmetric_test.csv
cp arithmetric_validation.csv arithmetric_test.csv



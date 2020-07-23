# CSV Detective with ML column types detection features

Installation : 

```
git clone git@github.com:etalab/csv_detective_ml.git
cd csv_detective_ml
mkdir tests/out
git clone git@github.com:etalab/csv_detective.git /tmp/csv_detective
mv /tmp/csv_detective/csv_detective .
conda create -f environment.yml
conda activate csv_deploy
```

Example to work (only with rule base for now) : 
```
python analyze_csv_cli.py tests/in/ tests/out/ YYYY-MM-DD --analysis_type=rule
```

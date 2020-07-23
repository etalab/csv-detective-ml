# CSV Detective with ML column types detection features

Installation : 

```
#Installation and preparation
git clone git@github.com:etalab/csv_detective_ml.git
cd csv_detective_ml
mkdir tests/out

# PATCH waiting for new version of csv-detective in pypi
git clone git@github.com:etalab/csv_detective.git /tmp/csv_detective
mv /tmp/csv_detective/csv_detective .

#Install env and activate it
conda create -f environment.yml
conda activate csv_deploy
```

Example to work (only with rule base for now) : 
```
python analyze_csv_cli.py tests/in/ tests/out/ YYYY-MM-DD --analysis_type=rule
```

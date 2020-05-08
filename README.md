# NeuralTune
Neural DBMS configuration tuner replacing the workload mapping and latency prediction steps from the baseline.

## Requirements  
pip install -r requirements.txt

## Data Loading
Please move the workload files to the .data/ folder within the NeuralTune directory. 

## Generating test_combined.csv  
After placing the files in the .data/ directory, to generate the test_combined.csv, call the genCombinedFile() function from file_utils.py

## Running Baseline: OtterTune Implementation
Run hyperparam_workload.py with the appropriate parameters.
For training run:   
python hyperparam_workload.py --method=baseline --mode=train --n_components=5 --length_scale=1 --output_variation=1 --noise=0.12 --dev_data=.data/online_workload_C.CSV  

For testing:
python hyperparam_workload.py --method=baseline --mode=test --n_components=5 --length_scale=1 --output_variation=1 --noise=0.12 --dev_data=.data/test_combined.CSV  

## Running Extensions: Topk
Run hyperparam_workload.py with the appropriate parameters.   
For training run:   
python hyperparam_workload.py --method=topk --mode=train --n_components=5 --length_scale=1 --output_variation=1 --noise=0.12 --dev_data=.data/online_workload_C.CSV  --topk=2

For testing:
python hyperparam_workload.py --method=topk --mode=test --n_components=5 --length_scale=1 --output_variation=1 --noise=0.12 --dev_data=.data/test_combined.CSV  --topk=2

## Running Extension: Threshold
Run hyperparam_workload.py with the appropriate parameters.   
For training run:   
python hyperparam_workload.py --method=threshold --mode=train --n_components=5 --length_scale=1 --output_variation=1 --noise=0.12 --dev_data=.data/online_workload_C.CSV  --threshold=4

For testing:
python hyperparam_workload.py --method=threshold --mode=test --n_components=5 --length_scale=1 --output_variation=1 --noise=0.12 --dev_data=.data/test_combined.CSV  --threshold=4


## Running Extension: NeuralTune

### Training:

1. Create a copy of `offline_workload.CSV`, name it `train.csv`, place it in `.data` directory. Similarly, create a copy of `online_workload_B.CSV` and name it `dev.csv`.

2. Update the `train_data_path` and `validation_data_path` in the `configs/best.json` to point to your created `.data` directory. 

4. Go to the **project root dir** and add it to the `PYTHONPATH`. This is so that the python interpretor can find the neuraltune package. On linux based systems using bash shell, this can be done using the following command:

```
export PYTHONPATH=`pwd`
```

3. Run the following command to begin training:

```
allennlp train configs/best.json -s best_model --include-package neuraltune -f
```

4. Run prediction on test set using the following command. This will create a file called `test_preds.json` in the `best_model` directory.

```
python neuraltune/predict.py --model_archive best_model/model.tar.gz --data_folder .data --scaler_path pruned_metrics1.pkl
```

**Please create an issue on [https://github.com/dhruvdcoder/neuraltune](https://github.com/dhruvdcoder/neuraltune) if the instructions are unclear**

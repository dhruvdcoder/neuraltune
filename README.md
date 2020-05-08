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




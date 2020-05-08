import numpy as np 
import itertools
import pandas as pd

def genCombinedFile():
    path = '.data/online_workload_C.csv'
    test = '.data/test.CSV'
  
    df_c = pd.read_csv(path)
    df_t = pd.read_csv(test)
    t = pd.concat([df_c,df_t], axis=0)
    t = t[df_c.columns]
    t.to_csv('.data/test_combined.csv',index=False)
   
def generate_test_predictions_csv():
    for method in ['baseline', 'topk', 'threshold']:
        print(method)
        a = np.load(f'test_{method}_transformed.npy', allow_pickle=True)
        df = pd.read_csv('.data/test.CSV')
        w = open(f'neighbors/{method}_neighbors.txt', 'w')
        w.write(f'workload_id, neighbors\n')
        for index, row in df.iterrows(): 
            for entry in a:
                if row['workload id'] == entry[-1]:
                    wid = row['workload id']
                    df.loc[index, "latency prediction"] = entry[0]
                    f = open(f'temp/{method}_{wid}.txt', 'r')
                    neighbors = f.read()
                    w.write(f'{wid}, {neighbors}\n')
                    f.close()
        df.to_csv(f'{method}_test.csv', index=False)
        f.close()
        w.close()

if __name__=='__main__':
    generate_test_predictions_csv()

from pyarrow import csv
import os,sys
import pysnooper

# @pysnooper.snoop()
def load(dataset_name):
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),f'{dataset_name}.csv')
    return csv.read_csv(csv_path)

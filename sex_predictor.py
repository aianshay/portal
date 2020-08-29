from joblib import load
import pandas as pd
import click



@click.command()
@click.option('--input_file', default=1, help='File name.')


def predict(input_file):    
    data = pd.read_csv(input_file)
    data = data.drop(columns=['nar', 'index', 'cp', 'slope'])
    
    clf = load('model.joblib')
    preds = clf.predict(data)
    

    
if __name__ == '__main__':
    predict()



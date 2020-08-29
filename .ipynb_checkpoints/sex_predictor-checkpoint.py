from joblib import load
import pandas as pd
import click
from sklearn.impute import SimpleImputer


@click.command()
@click.option('--input_file', help='File name.')


def predict(input_file):    
    data = pd.read_csv(input_file)
    data = data.drop(columns=['nar', 'index', 'cp', 'slope'])
    inputer = SimpleImputer(strategy='most_frequent')
    X_inputted = pd.DataFrame(columns=data.columns, data=inputer.fit_transform(data))
    
    clf = load('model.joblib')
    preds = clf.predict(X_inputted)
    preds_ = pd.DataFrame()
    preds_['sex'] = preds
    preds_.to_csv('newsample_predictions_aian_shay_cardoso.csv', index=False)
    
if __name__ == '__main__':
    predict()



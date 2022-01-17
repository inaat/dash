
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

from server import app, User
from flask_login import login_user
from werkzeug.security import check_password_hash


import cv2
import numpy as np
import tqdm.notebook as tqdm
import glob
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix as cm, classification_report as cr
import dash
from dash import dcc
from dash import html
import plotly.express as px
import pandas as pd
import dash_table

import base64
import io
import plotly.graph_objs as go





import dash
import dash_core_components as dcc
import dash_table as dt
import dash_html_components as html

from dash.dependencies import Output, Input
from dash.exceptions import PreventUpdate

import plotly.graph_objs as go

x = np.arange(1, 10)
benford = np.log10(1 + 1 / x)



layout = html.Div([
    html.H1('Deep Fake Detection'),
     dcc.Upload(
        id="upload-data",
        children=html.Div(["Drag and Drop or ", html.A("Select Files")]),
        style={
            "width": "100%",
            "height": "60px",
            "lineHeight": "60px",
            "borderWidth": "1px",
            "borderStyle": "dashed",
            "borderRadius": "5px",
            "textAlign": "center",
            "margin": "10px",
        },
        # Allow multiple files to be uploaded
        multiple=True,
    ),
    html.Div([
        dcc.Graph(id='graph'),
        html.H3('Deep Fake Detection Data Table'),
        dt.DataTable(id='data-table')
    ])
], id='container')


@app.callback([
    Output('graph', 'figure'),
    Output('data-table', 'data'),
    Output('data-table', 'columns'),
    Output('container', 'style')
], [
    Input('upload-data', 'contents'),
    Input('upload-data', 'filename')])
def multi_output(contents, filename):
   
    x = []
    y = []
    z=[]
    data = []
    columns=[]
    if contents:
        contents = contents[0]
        filename = filename[0]
        ima = parse_data(contents, filename)
        image = cv2.imread(ima)    
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        first_digits = compute_first_digits(img)
        unq, counts = np.unique(first_digits, return_counts=True)
        tot = counts.sum()
        counts = counts / tot
        df = pd.DataFrame()
        difference = np.abs(benford - counts)
        df['Picture Value'] = counts
        df['Benford Frequency'] = benford
        df['Unique'] = unq
        df['Difference'] = difference
        x=unq
        y=counts
        z=benford
        columns=[
            {'name': 'Unique', 'id': 'Unique'},
            {'name': 'Picture Value', 'id': 'Picture Value'},
            {'name': 'Benford Frequency', 'id': 'Benford Frequency'},
            {'name': 'Difference', 'id': 'Difference'}
        ]
        data=df.to_dict('records')

       
    figure = go.Figure()
    figure.add_trace(go.Line(x=x, y=y,
                    mode='lines+markers',
                    name=u'Picture Curve'
                    ))
   
    figure.add_trace(go.Line(x=x, y=z,
                    mode='lines+markers',
                    name=u'Benford Curve'
                    ))

    return figure, data, columns, {'display': 'block'}



def compute_first_digits(img, normalise=False, debug_dct=False):
    if isinstance(img, str):
        img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if normalise:
        norm = cv2.normalize(img, np.zeros(img.shape), 0, 255, cv2.NORM_MINMAX)

    dct = cv2.dct(np.float32(img) / 255.0)
    dct = np.abs(dct)  # Take abs values
    if debug_dct:
        print(dct)

    min_val = dct.min()
    if min_val < 1:
        dct = np.power(10, -np.floor(np.log10(min_val)) + 1) * \
            dct  # Scale all up to remove leading 0.00s

    if not (dct >= 1.0).all():
        raise ValueError("Error")

    digits = np.log10(dct).astype(int).astype('float32')
    first_digits = dct / np.power(10, digits)
    # Handle edge case.
    first_digits[(first_digits < 1.0) & (first_digits > 0.9)] = 1
    first_digits = first_digits.astype(int)

    if not (first_digits >= 1).all() and (first_digits <= 9).all():
        raise ValueError("Error")

    return first_digits


def parse_data(contents, filename):
    content_type, content_string = contents.split(",")

    decoded = base64.b64decode(content_string)
    with open("imageToTest.png", "wb") as fh:
        fh.write(decoded)

    return './imageToTest.png'
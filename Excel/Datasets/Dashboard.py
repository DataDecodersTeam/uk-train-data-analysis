import dash
from dash import dcc, html, callback, Input, Output
import pandas as pd
import plotly.express as px

# Load and preprocess data
df = pd.read_csv('D:\Data Analysis Course\UK Train Riders Data Analysis\Excel\Datasets\railway.csv')
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server  # For deployment

# Layout
app.layout = html.Div([
    html.H1("UK Train Riders Data Analysis Dashboard", style={'textAlign': 'center'}),

    html.Div([
        html.Label("Select Ticket Type"),
        dcc.Dropdown(
            id='ticket-type-filter',
            options=[{'label': ttype, 'value': ttype} for ttype in df['Ticket Type'].unique()],
            value='Advance',
            multi=False
        )
    ], style={'width': '40%', 'margin': 'auto'}),

    html.Br(),

    dcc.Tabs([
        dcc.Tab(label='Revenue by Ticket Type', children=[
            dcc.Graph(
                id='revenue-by-ticket-type',
                figure=px.bar(df.groupby('Ticket Type')['Price'].sum().reset_index(),
                              x='Ticket Type', y='Price', title='Total Revenue by Ticket Type')
            )
        ]),
        dcc.Tab(label='Journey Status Distribution', children=[
            dcc.Graph(
                id='journey-status-pie',
                figure=px.pie(df['Journey Status'].value_counts().reset_index(),
                              names='index', values='Journey Status', title='Journey Status Distribution')
            )
        ]),
        dcc.Tab(label='Transactions Over Time', children=[
            dcc.Graph(
                id='daily-transactions',
                figure={}
            )
        ])
    ])

])

# Callback example
@app.callback(
    Output('daily-transactions', 'figure'),
    Input('ticket-type-filter', 'value')
)
def update_daily_transactions(selected_ticket_type):
    filtered_df = df[df['Ticket Type'] == selected_ticket_type]
    daily_txns = filtered_df.groupby('Date of Purchase').size().reset_index(name='Count')
    fig = px.line(daily_txns, x='Date of Purchase', y='Count', title=f'Daily Transactions ({selected_ticket_type})')
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
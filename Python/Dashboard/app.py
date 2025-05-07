import calendar
import pandas as pd
import plotly.express as px
from dash import Dash, html, dcc, Input, Output, State, ctx
import dash_bootstrap_components as dbc

# Shared chart configurations
CHART_STYLE = {'height': '300px', 'border': '1px solid #dee2e6', 'box-shadow': '0 2px 5px rgba(0,0,0,0.05)'}
CHART_LAYOUT = {
    'plot_bgcolor': 'rgba(0,0,0,0)', 'yaxis': {'gridcolor': 'rgba(0,0,0,0.1)'},
    'font': {'size': 12}, 'title_x': 0.5, 'margin': {'l': 40, 'r': 40, 't': 40, 'b': 40}
}

# Load datasets with error handling
try:
    df_fact = pd.read_csv('fact_transactions.csv')
    df_journey = pd.read_csv('dim_journey.csv')
    df_location = pd.read_csv('dim_location.csv')
    df_time = pd.read_csv('dim_time.csv')
except FileNotFoundError:
    df_fact = pd.DataFrame()
    df_journey = pd.DataFrame()
    df_location = pd.DataFrame()
    df_time = pd.DataFrame()

# Clean column names
for df in [df_fact, df_journey, df_location, df_time]:
    df.columns = df.columns.str.strip()

# Optimize memory with categorical types
categorical_cols = ['Purchase_Type', 'Payment_Method', 'Railcard', 'Ticket_Class', 'Ticket_Type', 'Journey_Status', 'Refund_Request']
for col in categorical_cols:
    if col in df_fact.columns:
        df_fact[col] = df_fact[col].astype('category')
if 'Station_Name' in df_location.columns:
    df_location['Station_Name'] = df_location['Station_Name'].astype('category')

# Preprocess data
if not df_fact.empty:
    if 'Transaction_ID' in df_fact.columns:
        df_fact['Transaction_ID'] = df_fact['Transaction_ID'].astype(str)
    if 'Time_ID' in df_fact.columns:
        df_fact['Time_ID'] = df_fact['Time_ID'].astype(str)
    if not df_time.empty and 'Time_ID' in df_fact.columns and 'Time_ID' in df_time.columns:
        df_time['Time_ID'] = df_time['Time_ID'].astype(str)
        df_fact = df_fact.merge(df_time[['Time_ID', 'Month', 'Year', 'Purchase_Date', 'Hour_of_Day']], on='Time_ID', how='left')
    if not df_journey.empty and 'Journey_ID' in df_fact.columns:
        df_fact = df_fact.merge(df_journey[['Journey_ID', 'Journey_Date', 'Delay_Period', 'Reason_for_Delay']], on='Journey_ID', how='left')
    if not df_location.empty:
        if 'Departure_Station_ID' in df_fact.columns:
            df_fact = df_fact.merge(df_location[['Station_ID', 'Station_Name']], left_on='Departure_Station_ID', right_on='Station_ID', how='left').rename(columns={'Station_Name': 'Departure_Station_Name'}).drop(columns=['Station_ID'], errors='ignore')
        if 'Arrival_Station_ID' in df_fact.columns:
            df_fact = df_fact.merge(df_location[['Station_ID', 'Station_Name']], left_on='Arrival_Station_ID', right_on='Station_ID', how='left').rename(columns={'Station_Name': 'Arrival_Station_Name'}).drop(columns=['Station_ID'], errors='ignore')
    if 'Purchase_Date' in df_fact.columns:
        df_fact['Purchase_Date'] = pd.to_datetime(df_fact['Purchase_Date'], errors='coerce')
        df_fact['Month'] = df_fact['Purchase_Date'].dt.month
    if 'Journey_Date' in df_fact.columns:
        df_fact['Journey_Date'] = pd.to_datetime(df_fact['Journey_Date'], errors='coerce')

# Month options for dropdown
month_options = [{'label': calendar.month_name[m], 'value': m} for m in sorted(df_fact['Month'].unique()) if pd.notna(m)] if 'Month' in df_fact.columns else [{'label': 'No Data', 'value': 'no-data'}]

# Initialize Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "UK Train Rides Analysis"

# Define layout
app.layout = html.Div([
    # Top navigation bar
    html.Div([
        html.Div([
            html.Img(src='assets/UK-train.png', height="60px", style={'borderRadius': '50%', 'marginRight': '10px', 'objectFit': 'cover'}),
            html.H3("UK Train Rides Analysis", style={'margin': '0', 'color': '#2C3E50'})
        ], style={'width': '30%', 'padding': '10px', 'textAlign': 'left', 'display': 'flex', 'alignItems': 'center'}),
        html.Div([
            dbc.Nav(id='main-nav', children=[
                dbc.NavItem(dbc.NavLink("Overview", id="nav-overview", active=True, className="mx-1")),
                dbc.NavItem(dbc.NavLink("Revenue", id="nav-revenue", className="mx-1")),
                dbc.NavItem(dbc.NavLink("Journey", id="nav-journey", className="mx-1")),
                dbc.NavItem(dbc.NavLink("Performance", id="nav-performance", className="mx-1"))
            ], pills=True, justified=True)
        ], style={'width': '40%', 'textAlign': 'center'}),
        html.Div([
            html.Button("Filters", id="open-filters-btn", n_clicks=0, style={'padding': '8px 16px', 'fontSize': '16px'})
        ], style={'width': '30%', 'textAlign': 'right', 'padding': '10px'})
    ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'backgroundColor': '#F8F9F9', 'borderBottom': '1px solid #DDD', 'height': '60px'}),
    # Filters sidebar
    html.Div(id="filters-sidebar", className="sidebar", children=[
        html.Div([
            html.Button("×", id="close-filters-btn", style={'marginLeft': 'auto', 'fontSize': '20px', 'background': 'none', 'border': 'none'}),
            html.H5("Filters", style={'textAlign': 'center'}),
            html.Label("Month:", style={'marginTop': '20px'}),
            dcc.Dropdown(id='filter-month', options=month_options, placeholder="Select month", clearable=True),
            html.Label("Station Name:", style={'marginTop': '20px'}),
            dcc.Dropdown(id='filter-station', options=[{'label': s, 'value': s} for s in sorted(df_fact['Departure_Station_Name'].unique()) if pd.notna(s)] if 'Departure_Station_Name' in df_fact.columns else [], placeholder="Select station", clearable=True),
            html.Label("Ticket Type:", style={'marginTop': '20px'}),
            dcc.Dropdown(id='filter-ticket-type', options=[{'label': t, 'value': t} for t in df_fact['Ticket_Type'].unique() if pd.notna(t)] if 'Ticket_Type' in df_fact.columns else [], placeholder="Select ticket type", clearable=True),
            html.Label("Railcard:", style={'marginTop': '20px'}),
            dcc.Dropdown(id='filter-railcard', options=[{'label': r, 'value': r} for r in df_fact['Railcard'].unique() if pd.notna(r)] if 'Railcard' in df_fact.columns else [], placeholder="Select railcard", clearable=True),
            html.Label("Payment Method:", style={'marginTop': '20px'}),
            dcc.Dropdown(id='filter-payment', options=[{'label': p, 'value': p} for p in df_fact['Payment_Method'].unique() if pd.notna(p)] if 'Payment_Method' in df_fact.columns else [], placeholder="Select payment method", clearable=True)
        ], style={'padding': '20px'})
    ], style={'position': 'fixed', 'top': '60px', 'right': '-300px', 'width': '300px', 'height': 'calc(100vh - 60px)', 'backgroundColor': '#fff', 'boxShadow': '-2px 0 5px rgba(0,0,0,0.1)', 'transition': 'right 0.3s', 'zIndex': '1000', 'overflowY': 'auto'}),
    # Overlay for sidebar
    html.Div(id='overlay', style={'position': 'fixed', 'top': '60px', 'left': 0, 'right': 0, 'bottom': 0, 'backgroundColor': 'rgba(0,0,0,0.4)', 'display': 'none', 'zIndex': '999'}),
    # Dashboard sections
    html.Div(id='page-content', children=[
        html.Div([
            dbc.Row([dbc.Col(dcc.Graph(id='chart-transactions-hour', style=CHART_STYLE), width=8), dbc.Col(dcc.Graph(id='chart-revenue-ticket', style=CHART_STYLE), width=4)], className="mb-2"),
            dbc.Row([dbc.Col(dcc.Graph(id='chart-daily-transactions', style=CHART_STYLE), width=8), dbc.Col(dcc.Graph(id='chart-journey-status', style=CHART_STYLE), width=4)], className="mb-2")
        ], id='section-overview', className='dashboard-section'),
        html.Div([
            dbc.Row([dbc.Col(dcc.Graph(id='chart-daily-revenue', style=CHART_STYLE), width=12)], className="mb-2"),
            dbc.Row([dbc.Col(dcc.Graph(id='chart-ticket-class-revenue', style=CHART_STYLE), width=6), dbc.Col(dcc.Graph(id='chart-station-revenue', style=CHART_STYLE), width=6)], className="mb-2")
        ], id='section-revenue', className='dashboard-section'),
        html.Div([
            dbc.Row([dbc.Col(dcc.Graph(id='chart-delay-reasons', style=CHART_STYLE), width=6), dbc.Col(dcc.Graph(id='chart-railcard-usage', style=CHART_STYLE), width=6)], className="mb-2"),
            dbc.Row([dbc.Col(dcc.Graph(id='chart-avg-price-ticket', style=CHART_STYLE), width=6), dbc.Col(dcc.Graph(id='chart-purchase-type', style=CHART_STYLE), width=6)], className="mb-2")
        ], id='section-journey', className='dashboard-section'),
        html.Div([
            dbc.Row([dbc.Col(dcc.Graph(id='chart-revenue-refunded', style=CHART_STYLE), width=6), dbc.Col(dcc.Graph(id='chart-refunded-proportion', style=CHART_STYLE), width=6)], className="mb-2"),
            dbc.Row([dbc.Col(dcc.Graph(id='chart-refunded-count', style=CHART_STYLE), width=6), dbc.Col(dcc.Graph(id='chart-payment-method', style=CHART_STYLE), width=6)], className="mb-2")
        ], id='section-performance', className='dashboard-section')
    ], style={'padding': '10px', 'height': 'calc(100vh - 60px)', 'overflow': 'hidden', 'boxSizing': 'border-box'})
], style={'fontFamily': 'Arial, sans-serif', 'margin': '0', 'width': '100%', 'boxSizing': 'border-box'})

# Filter DataFrame based on user inputs
def filter_dataframe(month, station, ticket_type, railcard, payment):
    filtered_df = df_fact.copy()
    if month and 'Month' in filtered_df.columns and month != 'no-data':
        filtered_df = filtered_df[filtered_df['Month'] == month]
    if station and 'Departure_Station_Name' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Departure_Station_Name'] == station]
    if ticket_type and 'Ticket_Type' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Ticket_Type'] == ticket_type]
    if railcard and 'Railcard' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Railcard'] == railcard]
    if payment and 'Payment_Method' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Payment_Method'] == payment]
    return filtered_df

# Toggle filters sidebar
@app.callback(
    [Output('filters-sidebar', 'style'), Output('overlay', 'style')],
    [Input('open-filters-btn', 'n_clicks'), Input('close-filters-btn', 'n_clicks'), Input('overlay', 'n_clicks')],
    prevent_initial_call=True
)
def toggle_sidebar(open_clicks, close_clicks, overlay_clicks):
    base_style = {'position': 'fixed', 'top': '60px', 'width': '300px', 'height': 'calc(100vh - 60px)', 'backgroundColor': '#fff', 'boxShadow': '-2px 0 5px rgba(0,0,0,0.1)', 'zIndex': '1000'}
    if ctx.triggered_id == 'open-filters-btn':
        return [{**base_style, 'right': '0px'}, {'display': 'block'}]
    return [{**base_style, 'right': '-300px'}, {'display': 'none'}]

# Update navigation and section visibility
@app.callback(
    [Output('main-nav', 'children'), Output('section-overview', 'style'), Output('section-revenue', 'style'),
     Output('section-journey', 'style'), Output('section-performance', 'style')],
    [Input('nav-overview', 'n_clicks'), Input('nav-revenue', 'n_clicks'), Input('nav-journey', 'n_clicks'), Input('nav-performance', 'n_clicks')]
)
def update_section_visibility(*args):
    trigger_id = ctx.triggered_id or 'nav-overview'
    nav_items = [
        dbc.NavItem(dbc.NavLink("Overview", id="nav-overview", active=trigger_id == 'nav-overview', className="mx-1")),
        dbc.NavItem(dbc.NavLink("Revenue", id="nav-revenue", active=trigger_id == 'nav-revenue', className="mx-1")),
        dbc.NavItem(dbc.NavLink("Journey", id="nav-journey", active=trigger_id == 'nav-journey', className="mx-1")),
        dbc.NavItem(dbc.NavLink("Performance", id="nav-performance", active=trigger_id == 'nav-performance', className="mx-1"))
    ]
    visibility = {
        'nav-overview': [{'display': 'block'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}],
        'nav-revenue': [{'display': 'none'}, {'display': 'block'}, {'display': 'none'}, {'display': 'none'}],
        'nav-journey': [{'display': 'none'}, {'display': 'none'}, {'display': 'block'}, {'display': 'none'}],
        'nav-performance': [{'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'block'}]
    }
    return nav_items, *visibility.get(trigger_id, [{'display': 'block'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}])

# Update Overview charts
@app.callback(
    [Output('chart-transactions-hour', 'figure'), Output('chart-revenue-ticket', 'figure'),
     Output('chart-daily-transactions', 'figure'), Output('chart-journey-status', 'figure')],
    [Input('filter-month', 'value'), Input('filter-station', 'value'), Input('filter-ticket-type', 'value'),
     Input('filter-railcard', 'value'), Input('filter-payment', 'value')]
)
def update_overview_charts(month, station, ticket_type, railcard, payment):
    filtered_df = filter_dataframe(month, station, ticket_type, railcard, payment)
    empty_fig = px.scatter(x=[0], y=[0], title="No Data Available").update_traces(visible=False)
    if filtered_df.empty:
        return empty_fig, empty_fig, empty_fig, empty_fig

    # Transactions by Hour
    transactions_hour = filtered_df.groupby('Hour_of_Day', observed=False).size().reset_index(name='Number of Transactions') if 'Hour_of_Day' in filtered_df.columns else pd.DataFrame({'Hour_of_Day': range(24), 'Number of Transactions': [0]*24})
    fig1 = px.line(transactions_hour, x='Hour_of_Day', y='Number of Transactions', title='Number of Transactions by Hour of Day', markers=True, line_shape='linear', template='plotly_white')
    fig1.update_layout(**CHART_LAYOUT, xaxis_title='Hour of Day', yaxis_title='Number of Transactions', showlegend=False, xaxis={'tickmode': 'linear', 'tick0': 0, 'dtick': 1})

    # Revenue by Ticket Type
    ticket_type_revenue = filtered_df.groupby('Ticket_Type', observed=False)['Price'].sum().reset_index() if 'Ticket_Type' in filtered_df.columns and 'Price' in filtered_df.columns else pd.DataFrame({'Ticket_Type': [], 'Price': []})
    fig2 = px.bar(ticket_type_revenue, x='Ticket_Type', y='Price', title='Revenue by Ticket Type', template='plotly_white')
    fig2.update_layout(**CHART_LAYOUT, xaxis_title='Ticket Type', yaxis_title='Revenue ($)')

    # Daily Transactions
    daily_transactions = filtered_df.groupby('Purchase_Date', observed=False).size().reset_index(name='Number of Transactions') if 'Purchase_Date' in filtered_df.columns else pd.DataFrame({'Purchase_Date': [], 'Number of Transactions': []})
    fig3 = px.line(daily_transactions, x='Purchase_Date', y='Number of Transactions', title='Daily Number of Transactions', template='plotly_white')
    fig3.update_layout(**CHART_LAYOUT, xaxis_title='Date', yaxis_title='Number of Transactions', xaxis_tickformat='%b %d')

    # Journey Status Distribution
    journey_status_dist = filtered_df['Journey_Status'].value_counts().reset_index(name='Count') if 'Journey_Status' in filtered_df.columns else pd.DataFrame({'Journey_Status': [], 'Count': []})
    fig4 = px.pie(journey_status_dist, names='Journey_Status', values='Count', title='Journey Status Distribution', hole=0.5, template='plotly_white')
    fig4.update_traces(textinfo='percent+label', pull=[0.05]*len(journey_status_dist))
    # Merge layout parameters to avoid margin conflict
    fig4.update_layout(**{**CHART_LAYOUT, 'showlegend': True, 'margin': {'l': 10, 'r': 10, 't': 80, 'b': 10}})

    return fig1, fig2, fig3, fig4

# Update Revenue charts
@app.callback(
    [Output('chart-daily-revenue', 'figure'), Output('chart-ticket-class-revenue', 'figure'), Output('chart-station-revenue', 'figure')],
    [Input('filter-month', 'value'), Input('filter-station', 'value'), Input('filter-ticket-type', 'value'),
     Input('filter-railcard', 'value'), Input('filter-payment', 'value')]
)
def update_revenue_charts(month, station, ticket_type, railcard, payment):
    filtered_df = filter_dataframe(month, station, ticket_type, railcard, payment)
    empty_fig = px.scatter(x=[0], y=[0], title="No Data Available").update_traces(visible=False)
    if filtered_df.empty:
        return empty_fig, empty_fig, empty_fig

    # Daily Revenue
    daily_revenue = filtered_df.groupby('Journey_Date', observed=False)['Price'].sum().reset_index(name='Daily Revenue') if 'Journey_Date' in filtered_df.columns and 'Price' in filtered_df.columns else pd.DataFrame({'Journey_Date': [], 'Daily Revenue': []})
    fig1 = px.line(daily_revenue, x='Journey_Date', y='Daily Revenue', title='Daily Revenue', template='plotly_white')
    fig1.update_layout(**CHART_LAYOUT, xaxis_title='Date', yaxis_title='Revenue ($)', xaxis_tickformat='%b %d', showlegend=False)

    # Revenue by Ticket Class
    ticket_class_revenue = filtered_df.groupby('Ticket_Class', observed=False)['Price'].sum().reset_index() if 'Ticket_Class' in filtered_df.columns and 'Price' in filtered_df.columns else pd.DataFrame({'Ticket_Class': [], 'Price': []})
    fig2 = px.pie(ticket_class_revenue, names='Ticket_Class', values='Price', title='Revenue Distribution by Ticket Class', template='plotly_white')
    fig2.update_traces(textinfo='percent+label').update_layout(**CHART_LAYOUT, showlegend=True)

    # Revenue by Departure Station
    station_revenue = filtered_df.groupby('Departure_Station_Name', observed=False)['Price'].sum().sort_values(ascending=False).head(5).reset_index() if 'Departure_Station_Name' in filtered_df.columns and 'Price' in filtered_df.columns else pd.DataFrame({'Departure_Station_Name': [], 'Price': []})
    fig3 = px.bar(station_revenue, x='Price', y='Departure_Station_Name', title='Revenue by Departure Station', template='plotly_white')
    fig3.update_layout(**CHART_LAYOUT, xaxis_title='Revenue', yaxis_title='Station')

    return fig1, fig2, fig3

# Update Journey charts
@app.callback(
    [Output('chart-delay-reasons', 'figure'), Output('chart-railcard-usage', 'figure'),
     Output('chart-avg-price-ticket', 'figure'), Output('chart-purchase-type', 'figure')],
    [Input('filter-month', 'value'), Input('filter-station', 'value'), Input('filter-ticket-type', 'value'),
     Input('filter-railcard', 'value'), Input('filter-payment', 'value')]
)
def update_journey_charts(month, station, ticket_type, railcard, payment):
    filtered_df = filter_dataframe(month, station, ticket_type, railcard, payment)
    empty_fig = px.scatter(x=[0], y=[0], title="No Data Available").update_traces(visible=False)
    if filtered_df.empty:
        return empty_fig, empty_fig, empty_fig, empty_fig

    # Delay Reasons
    delay_reasons = filtered_df[filtered_df['Reason_for_Delay'] != 'No Delay']['Reason_for_Delay'].value_counts().reset_index() if 'Reason_for_Delay' in filtered_df.columns else pd.DataFrame({'Reason': [], 'Count': []})
    delay_reasons.columns = ['Reason', 'Count']
    fig1 = px.bar(delay_reasons, x='Count', y='Reason', title='Delay Reasons', template='plotly_white')
    fig1.update_layout(**CHART_LAYOUT, xaxis_title='Count', yaxis_title='Reason')

    # Railcard Usage
    railcard_usage = filtered_df['Railcard'].value_counts().reset_index() if 'Railcard' in filtered_df.columns else pd.DataFrame({'Railcard': [], 'Number of Transactions': []})
    railcard_usage.columns = ['Railcard', 'Number of Transactions']
    fig2 = px.bar(railcard_usage, x='Railcard', y='Number of Transactions', title='Railcard Usage', template='plotly_white')
    fig2.update_layout(**CHART_LAYOUT, xaxis_title='Railcard Type', yaxis_title='Number of Transactions')

    # Average Price by Ticket Type
    avg_price_by_ticket = filtered_df.groupby('Ticket_Type', observed=False)['Price'].mean().reset_index() if 'Ticket_Type' in filtered_df.columns and 'Price' in filtered_df.columns else pd.DataFrame({'Ticket_Type': [], 'Price': []})
    fig3 = px.bar(avg_price_by_ticket, x='Ticket_Type', y='Price', title='Average Price by Ticket Type', template='plotly_white')
    fig3.update_layout(**CHART_LAYOUT, xaxis_title='Ticket Type', yaxis_title='Average Price ($)')

    # Purchase Type Distribution
    purchase_type_counts = filtered_df['Purchase_Type'].value_counts().reset_index(name='Count') if 'Purchase_Type' in filtered_df.columns else pd.DataFrame({'Purchase_Type': [], 'Count': []})
    fig4 = px.pie(purchase_type_counts, names='Purchase_Type', values='Count', title='Number of Transactions by Purchase Type', template='plotly_white')
    fig4.update_traces(textinfo='percent+label').update_layout(**CHART_LAYOUT, showlegend=True)

    return fig1, fig2, fig3, fig4

# Update Performance charts
@app.callback(
    [Output('chart-revenue-refunded', 'figure'), Output('chart-refunded-proportion', 'figure'),
     Output('chart-refunded-count', 'figure'), Output('chart-payment-method', 'figure')],
    [Input('filter-month', 'value'), Input('filter-station', 'value'), Input('filter-ticket-type', 'value'),
     Input('filter-railcard', 'value'), Input('filter-payment', 'value')]
)
def update_performance_charts(month, station, ticket_type, railcard, payment):
    filtered_df = filter_dataframe(month, station, ticket_type, railcard, payment)
    empty_fig = px.scatter(x=[0], y=[0], title="No Data Available").update_traces(visible=False)
    if filtered_df.empty:
        return empty_fig, empty_fig, empty_fig, empty_fig

    # Revenue by Journey Status and Refund
    revenue_refunded = filtered_df.groupby(['Journey_Status', 'Refund_Request'], observed=False)['Price'].sum().reset_index() if 'Journey_Status' in filtered_df.columns and 'Refund_Request' in filtered_df.columns and 'Price' in filtered_df.columns else pd.DataFrame()
    fig1 = px.bar(revenue_refunded, x='Journey_Status', y='Price', color='Refund_Request', barmode='group', title='Revenue by Journey Status and Refund Request') if not revenue_refunded.empty else empty_fig
    fig1.update_layout(**CHART_LAYOUT, xaxis_title='Journey Status', yaxis_title='Revenue ($)', legend_title_text='Refund Requested')

    # Refund Proportion
    refund_proportion = filtered_df['Refund_Request'].value_counts().reset_index(name='Count') if 'Refund_Request' in filtered_df.columns else pd.DataFrame()
    fig2 = px.pie(refund_proportion, names='Refund_Request', values='Count', title='Proportion of Refund Requests') if not refund_proportion.empty else empty_fig
    fig2.update_traces(textinfo='percent+label')
    # Merge layout parameters to avoid margin conflict
    fig2.update_layout(**{**CHART_LAYOUT, 'showlegend': True, 'margin': {'l': 40, 'r': 40, 't': 70, 'b': 10}})

    # Refund Requests by Journey Status
    refund_count = filtered_df.groupby(['Journey_Status', 'Refund_Request'], observed=False).size().reset_index(name='Count') if 'Journey_Status' in filtered_df.columns and 'Refund_Request' in filtered_df.columns else pd.DataFrame()
    fig3 = px.bar(refund_count, x='Count', y='Journey_Status', color='Refund_Request', barmode='group', title='Refund Requests by Journey Status') if not refund_count.empty else empty_fig
    fig3.update_layout(**CHART_LAYOUT, xaxis_title='Number of Transactions', yaxis_title='Journey Status', legend_title_text='Refund Requested')

    # Payment Method Distribution
    payment_method_dist = filtered_df['Payment_Method'].value_counts().reset_index(name='Count') if 'Payment_Method' in filtered_df.columns else pd.DataFrame()
    fig4 = px.pie(payment_method_dist, names='Payment_Method', values='Count', title='Payment Method Distribution') if not payment_method_dist.empty else empty_fig
    fig4.update_traces(textinfo='percent+label').update_layout(**CHART_LAYOUT, showlegend=True)

    return fig1, fig2, fig3, fig4

# Run the server
if __name__ == '__main__':
    app.run(debug=True)
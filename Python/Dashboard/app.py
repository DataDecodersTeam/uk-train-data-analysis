import dash
from dash import html, dcc, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import pandas as pd
import calendar
import plotly.express as px
import plotly.graph_objects as go

# Load Datasets with Error Handling
try:
    df_fact = pd.read_csv('fact_transactions.csv')
    df_journey = pd.read_csv('dim_journey.csv')
    df_location = pd.read_csv('dim_location.csv')
    df_time = pd.read_csv('dim_time.csv')
except FileNotFoundError as e:
    print(f"Error: {e}")
    df_fact = pd.DataFrame()
    df_journey = pd.DataFrame()
    df_location = pd.DataFrame()
    df_time = pd.DataFrame()

# Strip whitespace from column names
df_fact.columns = df_fact.columns.str.strip()
df_journey.columns = df_journey.columns.str.strip()
df_location.columns = df_location.columns.str.strip()
df_time.columns = df_time.columns.str.strip()

# Print columns and Time_ID types for debugging
print("fact_transactions columns:", df_fact.columns.tolist())
print("dim_journey columns:", df_journey.columns.tolist())
print("dim_location columns:", df_location.columns.tolist())
print("dim_time columns:", df_time.columns.tolist())
if 'Time_ID' in df_fact.columns:
    print("fact_transactions Time_ID dtype:", df_fact['Time_ID'].dtype)
if 'Time_ID' in df_time.columns:
    print("dim_time Time_ID dtype:", df_time['Time_ID'].dtype)
if 'Railcard' in df_fact.columns:
    print("Unique Railcard values:", df_fact['Railcard'].unique().tolist())

# Check if df_fact is empty
if df_fact.empty:
    print("Warning: df_fact is empty. Check if CSV files exist and are correctly formatted.")

# Optimize memory for categorical columns
categorical_columns = ['Purchase_Type', 'Payment_Method', 'Railcard', 'Ticket_Class', 'Ticket_Type', 'Journey_Status', 'Refund_Request']
for col in categorical_columns:
    if col in df_fact.columns:
        try:
            df_fact[col] = df_fact[col].astype('category')
        except Exception as e:
            print(f"Warning: Could not convert {col} to category: {e}")
if 'Station_Name' in df_location.columns:
    df_location['Station_Name'] = df_location['Station_Name'].astype('category')

# Preprocess data
if not df_fact.empty:
    # Ensure Transaction_ID and Time_ID are strings
    if 'Transaction_ID' in df_fact.columns:
        df_fact['Transaction_ID'] = df_fact['Transaction_ID'].astype(str)
    if 'Time_ID' in df_fact.columns:
        df_fact['Time_ID'] = df_fact['Time_ID'].astype(str)
        print("fact_transactions Time_ID dtype after conversion:", df_fact['Time_ID'].dtype)
    
    # Merge with dim_time to get Purchase_Date and Hour_of_Day
    if not df_time.empty and 'Time_ID' in df_fact.columns and 'Time_ID' in df_time.columns:
        df_time['Time_ID'] = df_time['Time_ID'].astype(str)
        print("dim_time Time_ID dtype after conversion:", df_time['Time_ID'].dtype)
        df_fact = df_fact.merge(
            df_time[['Time_ID', 'Month', 'Year', 'Purchase_Date', 'Hour_of_Day']],
            on='Time_ID',
            how='left'
        )
        print("After dim_time merge, df_fact shape:", df_fact.shape)
        print("Sample Purchase_Date:", df_fact['Purchase_Date'].head().tolist())
        print("Sample Hour_of_Day:", df_fact['Hour_of_Day'].head().tolist())
    else:
        print("Warning: dim_time merge skipped; Time_ID not found or df_time is empty.")
    
    # Merge with dim_journey
    if not df_journey.empty and 'Journey_ID' in df_fact.columns:
        df_fact = df_fact.merge(
            df_journey[['Journey_ID', 'Journey_Date', 'Delay_Period', 'Reason_for_Delay']],
            on='Journey_ID',
            how='left'
        )
        print("After dim_journey merge, df_fact shape:", df_fact.shape)
        print("df_fact columns after dim_journey merge:", df_fact.columns.tolist())
    
    # Merge with dim_location for Departure and Arrival stations
    if not df_location.empty and 'Departure_Station_ID' in df_fact.columns:
        df_fact = df_fact.merge(
            df_location[['Station_ID', 'Station_Name']],
            left_on='Departure_Station_ID',
            right_on='Station_ID',
            how='left'
        ).rename(columns={'Station_Name': 'Departure_Station_Name'}).drop(columns=['Station_ID'], errors='ignore')
        print("After departure dim_location merge, df_fact shape:", df_fact.shape)
    
    if not df_location.empty and 'Arrival_Station_ID' in df_fact.columns:
        df_fact = df_fact.merge(
            df_location[['Station_ID', 'Station_Name']],
            left_on='Arrival_Station_ID',
            right_on='Station_ID',
            how='left'
        ).rename(columns={'Station_Name': 'Arrival_Station_Name'}).drop(columns=['Station_ID'], errors='ignore')
        print("After arrival dim_location merge, df_fact shape:", df_fact.shape)
        print("df_fact columns after all merges:", df_fact.columns.tolist())

    # Ensure Purchase_Date and Journey_Date are datetime
    if 'Purchase_Date' in df_fact.columns:
        invalid_dates = df_fact['Purchase_Date'][df_fact['Purchase_Date'].isna()]
        if not invalid_dates.empty:
            print("Warning: Invalid Purchase_Date values found:", invalid_dates.head().tolist())
        df_fact['Purchase_Date'] = pd.to_datetime(df_fact['Purchase_Date'], errors='coerce')
        df_fact['Month'] = df_fact['Purchase_Date'].dt.month
        print("Sample Purchase_Date after datetime conversion:", df_fact['Purchase_Date'].head().tolist())
    
    if 'Journey_Date' in df_fact.columns:
        invalid_journey_dates = df_fact['Journey_Date'][df_fact['Journey_Date'].isna()]
        if not invalid_journey_dates.empty:
            print("Warning: Invalid Journey_Date values found:", invalid_journey_dates.head().tolist())
        df_fact['Journey_Date'] = pd.to_datetime(df_fact['Journey_Date'], errors='coerce')
        print("Sample Journey_Date after datetime conversion:", df_fact['Journey_Date'].head().tolist())

# Month options for dropdown
month_options = [{'label': calendar.month_name[m], 'value': m} for m in sorted(df_fact['Month'].unique()) if pd.notna(m)] if 'Month' in df_fact.columns else [{'label': 'No Data', 'value': 'no-data'}]

# Initialize Dash App
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "UK Train Rides Analysis"

# App Layout
app.layout = html.Div([
    # Top Navigation Bar
    html.Div([
        html.Div([
            html.Img(
                src='assets/UK-train.png',
                height="60px",
                style={'borderRadius': '50%', 'marginRight': '10px', 'objectFit': 'cover'}
            ),
            html.H3("UK Train Rides Analysis", style={'margin': '0', 'color': '#2C3E50'})
        ], style={
            'width': '30%',
            'padding': '10px',
            'textAlign': 'left',
            'display': 'flex',
            'alignItems': 'center'
        }),
        html.Div([
            dbc.Nav(id='main-nav', children=[
                dbc.NavItem(dbc.NavLink("Overview", id="nav-overview", active=True, className="mx-1")),
                dbc.NavItem(dbc.NavLink("Revenue", id="nav-revenue", className="mx-1")),
                dbc.NavItem(dbc.NavLink("Journey", id="nav-journey", className="mx-1")),
                dbc.NavItem(dbc.NavLink("Performance", id="nav-performance", className="mx-1"))
            ], pills=True, justified=True)
        ], style={'width': '40%', 'textAlign': 'center'}),
        html.Div([
            html.Button("Filters", id="open-filters-btn", n_clicks=0,
                        style={'padding': '8px 16px', 'fontSize': '16px'})
        ], style={'width': '30%', 'textAlign': 'right', 'padding': '10px'})
    ], style={
        'display': 'flex',
        'justifyContent': 'space-between',
        'alignItems': 'center',
        'backgroundColor': '#F8F9F9',
        'borderBottom': '1px solid #DDD',
        'height': '60px'
    }),

    # Sidebar for Filters
    html.Div(id="filters-sidebar", className="sidebar", children=[
        html.Div([
            html.Button("×", id="close-filters-btn", style={
                'marginLeft': 'auto',
                'fontSize': '20px',
                'background': 'none',
                'border': 'none'
            }),
            html.H5("Filters", style={'textAlign': 'center'}),
            html.Label("Month:", style={'marginTop': '20px'}),
            dcc.Dropdown(
                id='filter-month',
                options=month_options,
                placeholder="Select month",
                clearable=True
            ),
            html.Label("Station Name:", style={'marginTop': '20px'}),
            dcc.Dropdown(
                id='filter-station',
                options=[{'label': s, 'value': s} for s in sorted(df_fact['Departure_Station_Name'].unique()) if pd.notna(s)] if 'Departure_Station_Name' in df_fact.columns else [],
                placeholder="Select station",
                clearable=True
            ),
            html.Label("Ticket Type:", style={'marginTop': '20px'}),
            dcc.Dropdown(
                id='filter-ticket-type',
                options=[{'label': t, 'value': t} for t in df_fact['Ticket_Type'].unique() if pd.notna(t)] if 'Ticket_Type' in df_fact.columns else [],
                placeholder="Select ticket type",
                clearable=True
            ),
            html.Label("Railcard:", style={'marginTop': '20px'}),
            dcc.Dropdown(
                id='filter-railcard',
                options=[{'label': r, 'value': r} for r in df_fact['Railcard'].unique() if pd.notna(r)] if 'Railcard' in df_fact.columns else [],
                placeholder="Select railcard",
                clearable=True
            ),
            html.Label("Payment Method:", style={'marginTop': '20px'}),
            dcc.Dropdown(
                id='filter-payment',
                options=[{'label': p, 'value': p} for p in df_fact['Payment_Method'].unique() if pd.notna(p)] if 'Payment_Method' in df_fact.columns else [],
                placeholder="Select payment method",
                clearable=True
            ),
        ], style={'padding': '20px'})
    ], style={
        'position': 'fixed',
        'top': '60px',
        'right': '-300px',
        'width': '300px',
        'height': 'calc(100vh - 60px)',
        'backgroundColor': '#fff',
        'boxShadow': '-2px 0 5px rgba(0,0,0,0.1)',
        'transition': 'right 0.3s',
        'zIndex': '1000',
        'overflowY': 'auto'
    }),

    # Overlay to close sidebar
    html.Div(id='overlay', style={
        'position': 'fixed',
        'top': '60px',
        'left': 0,
        'right': 0,
        'bottom': 0,
        'backgroundColor': 'rgba(0,0,0,0.4)',
        'display': 'none',
        'zIndex': '999'
    }),

    # Dashboard Sections
    html.Div(id='page-content', children=[
        # Overview Section
        html.Div([
            # Row 1: Transactions by Hour, Revenue by Ticket Type
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='chart-transactions-hour', style={'height': '300px', 'border': '1px solid #dee2e6', 'box-shadow': '0 2px 5px rgba(0, 0, 0, 0.05)'})
                ], width=8),
                dbc.Col([
                    dcc.Graph(id='chart-revenue-ticket', style={'height': '300px', 'border': '1px solid #dee2e6', 'box-shadow': '0 2px 5px rgba(0, 0, 0, 0.05)'})
                ], width=4)
            ], className="mb-2"),
            # Row 2: Daily Transactions, Journey Status Distribution
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='chart-daily-transactions', style={'height': '300px', 'border': '1px solid #dee2e6', 'box-shadow': '0 2px 5px rgba(0, 0, 0, 0.05)'})
                ], width=8),
                dbc.Col([
                    dcc.Graph(id='chart-journey-status', style={'height': '300px', 'border': '1px solid #dee2e6', 'box-shadow': '0 2px 5px rgba(0, 0, 0, 0.05)'})
                ], width=4)
            ], className="mb-2")
        ], id='section-overview', className='dashboard-section', style={'height': '300px', 'border': '1px solid #dee2e6', 'box-shadow': '0 2px 5px rgba(0, 0, 0, 0.05)'}),

        # Revenue Section
        html.Div([
            # Row 1: Daily Revenue
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='chart-daily-revenue', style={'height': '300px', 'border': '1px solid #dee2e6', 'box-shadow': '0 2px 5px rgba(0, 0, 0, 0.05)'})
                ], width=12)
            ], className="mb-2"),
            # Row 2: Ticket Class Revenue, Station Revenue
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='chart-ticket-class-revenue', style={'height': '300px', 'border': '1px solid #dee2e6', 'box-shadow': '0 2px 5px rgba(0, 0, 0, 0.05)'})
                ], width=6),
                dbc.Col([
                    dcc.Graph(id='chart-station-revenue', style={'height': '300px', 'border': '1px solid #dee2e6', 'box-shadow': '0 2px 5px rgba(0, 0, 0, 0.05)'})
                ], width=6)
            ], className="mb-2")
        ], id='section-revenue', className='dashboard-section', style={'height': '300px', 'border': '1px solid #dee2e6', 'box-shadow': '0 2px 5px rgba(0, 0, 0, 0.05)'}),

        # Journey Section
        html.Div([
            # Row 1: Delay Reasons, Railcard Usage
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='chart-delay-reasons', style={'height': '300px', 'border': '1px solid #dee2e6', 'box-shadow': '0 2px 5px rgba(0, 0, 0, 0.05)'})
                ], width=6),
                dbc.Col([
                    dcc.Graph(id='chart-railcard-usage', style={'height': '300px', 'border': '1px solid #dee2e6', 'box-shadow': '0 2px 5px rgba(0, 0, 0, 0.05)'})
                ], width=6)
            ], className="mb-2"),
            # Row 2: Average Price by Ticket Type, Purchase Type Distribution
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='chart-avg-price-ticket', style={'height': '300px', 'border': '1px solid #dee2e6', 'box-shadow': '0 2px 5px rgba(0, 0, 0, 0.05)'})
                ], width=6),
                dbc.Col([
                    dcc.Graph(id='chart-purchase-type', style={'height': '300px', 'border': '1px solid #dee2e6', 'box-shadow': '0 2px 5px rgba(0, 0, 0, 0.05)'})
                ], width=6)
            ], className="mb-2")
        ], id='section-journey', className='dashboard-section'),

        # Performance Section
        html.Div([
            # Row 1: Revenue Impact of Refund Requests, Refund Request Proportion
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='chart-revenue-refunded', style={'height': '300px', 'border': '1px solid #dee2e6', 'box-shadow': '0 2px 5px rgba(0, 0, 0, 0.05)'})
                ], width=6),
                dbc.Col([
                    dcc.Graph(id='chart-refunded-proportion', style={'height': '300px', 'border': '1px solid #dee2e6', 'box-shadow': '0 2px 5px rgba(0, 0, 0, 0.05)'})
                ], width=6)
            ], className="mb-2"),
            # Row 2: Refund Requests by Journey Status, Payment Method Distribution
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='chart-refunded-count', style={'height': '300px', 'border': '1px solid #dee2e6', 'box-shadow': '0 2px 5px rgba(0, 0, 0, 0.05)'})
                ], width=6),
                dbc.Col([
                    dcc.Graph(id='chart-payment-method', style={'height': '300px', 'border': '1px solid #dee2e6', 'box-shadow': '0 2px 5px rgba(0, 0, 0, 0.05)'})
                ], width=6)
            ], className="mb-2")
        ], id='section-performance', className='dashboard-section'),
    ], style={
        'padding': '10px',
        'height': 'calc(100vh - 60px)',
        'overflow': 'hidden',
        'boxSizing': 'border-box'
    })
], style={
    'fontFamily': 'Arial, sans-serif',
    'margin': '0',
    'width': '100%',
    'boxSizing': 'border-box'
})

# --- Callbacks ---

# Toggle Filters Sidebar
@app.callback(
    [Output('filters-sidebar', 'style'), Output('overlay', 'style')],
    [Input('open-filters-btn', 'n_clicks'),
     Input('close-filters-btn', 'n_clicks'),
     Input('overlay', 'n_clicks')],
    prevent_initial_call=True
)
def toggle_sidebar(open_clicks, close_clicks, overlay_clicks):
    trigger_id = ctx.triggered_id
    if trigger_id == 'open-filters-btn':
        return (
            {'right': '0px', 'position': 'fixed', 'top': '60px', 'width': '300px', 'height': 'calc(100vh - 60px)',
             'backgroundColor': '#fff', 'boxShadow': '-2px 0 5px rgba(0,0,0,0.1)', 'zIndex': '1000'},
            {'display': 'block'}
        )
    else:
        return (
            {'right': '-300px', 'position': 'fixed', 'title': 'app.py', 'width': '300px', 'height': 'calc(100vh - 60px)',
             'backgroundColor': '#fff', 'boxShadow': '-2px 0 5px rgba(0,0,0,0.1)', 'zIndex': '1000'},
            {'display': 'none'}
        )

# Update active nav item and section visibility
@app.callback(
    [Output('main-nav', 'children'),
     Output('section-overview', 'style'),
     Output('section-revenue', 'style'),
     Output('section-journey', 'style'),
     Output('section-performance', 'style')],
    [Input('nav-overview', 'n_clicks'),
     Input('nav-revenue', 'n_clicks'),
     Input('nav-journey', 'n_clicks'),
     Input('nav-performance', 'n_clicks')],
)
def update_section_visibility(*args):
    trigger_id = ctx.triggered_id or 'nav-overview'
    nav_items = [
        dbc.NavItem(dbc.NavLink("Overview", id="nav-overview", active=(trigger_id == 'nav-overview'), className="mx-1")),
        dbc.NavItem(dbc.NavLink("Revenue", id="nav-revenue", active=(trigger_id == 'nav-revenue'), className="mx-1")),
        dbc.NavItem(dbc.NavLink("Journey", id="nav-journey", active=(trigger_id == 'nav-journey'), className="mx-1")),
        dbc.NavItem(dbc.NavLink("Performance", id="nav-performance", active=(trigger_id == 'nav-performance'), className="mx-1")),
    ]
    visibility = {
        'nav-overview': [{'display': 'block'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}],
        'nav-revenue': [{'display': 'none'}, {'display': 'block'}, {'display': 'none'}, {'display': 'none'}],
        'nav-journey': [{'display': 'none'}, {'display': 'none'}, {'display': 'block'}, {'display': 'none'}],
        'nav-performance': [{'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'block'}],
    }
    return nav_items, *visibility.get(trigger_id, [{'display': 'block'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}])

# Update Overview Charts
@app.callback(
    [
        Output('chart-transactions-hour', 'figure'),
        Output('chart-revenue-ticket', 'figure'),
        Output('chart-daily-transactions', 'figure'),
        Output('chart-journey-status', 'figure')
    ],
    [
        Input('filter-month', 'value'),
        Input('filter-station', 'value'),
        Input('filter-ticket-type', 'value'),
        Input('filter-railcard', 'value'),
        Input('filter-payment', 'value')
    ]
)
def update_overview_charts(month, station, ticket_type, railcard, payment):
    # Filter dataframe
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

    # Debug: Print filtered dataframe info
    print(f"Filtered dataframe shape: {filtered_df.shape}")
    print(f"Filtered dataframe columns: {filtered_df.columns.tolist()}")
    if not filtered_df.empty:
        if 'Purchase_Date' in filtered_df.columns:
            print(f"Sample Purchase_Date: {filtered_df['Purchase_Date'].head().tolist()}")
        if 'Hour_of_Day' in filtered_df.columns:
            print(f"Sample Hour_of_Day: {filtered_df['Hour_of_Day'].head().tolist()}")

    # Handle empty dataframe
    if filtered_df.empty:
        empty_fig = px.scatter(x=[0], y=[0], title="No Data Available").update_traces(visible=False)
        return empty_fig, empty_fig, empty_fig, empty_fig

    # Chart 1: Transactions by Hour of Day
    if 'Hour_of_Day' in filtered_df.columns:
        transactions_hour = filtered_df.groupby('Hour_of_Day', observed=False).size().reset_index(name='Number of Transactions')
    else:
        transactions_hour = pd.DataFrame({'Hour_of_Day': range(24), 'Number of Transactions': [0]*24})
        print("Warning: Hour_of_Day not in filtered_df; using fallback data.")
    fig1 = px.line(
        transactions_hour,
        x='Hour_of_Day',
        y='Number of Transactions',
        title='Number of Transactions by Hour of Day',
        markers=True,
        line_shape='linear',
        template='plotly_white'
    )
    fig1.update_layout(
        xaxis_title='Hour of Day',
        yaxis_title='Number of Transactions',
        showlegend=False,
        xaxis=dict(tickmode='linear', tick0=0, dtick=1),
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
        font=dict(size=12),
        title_x=0.5,
        margin=dict(l=40, r=40, t=40, b=40)
    )

    # Chart 2: Revenue by Ticket Type
    if 'Ticket_Type' in filtered_df.columns and 'Price' in filtered_df.columns:
        ticket_type_revenue = filtered_df.groupby('Ticket_Type', observed=False)['Price'].sum().reset_index()
    else:
        ticket_type_revenue = pd.DataFrame({'Ticket_Type': [], 'Price': []})
        print("Warning: Ticket_Type or Price not in filtered_df.")
    fig2 = px.bar(
        ticket_type_revenue,
        x='Ticket_Type',
        y='Price',
        title='Revenue by Ticket Type',
        template='plotly_white'
    )
    fig2.update_layout(
        xaxis_title='Ticket Type',
        yaxis_title='Revenue ($)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
        font=dict(size=12),
        title_x=0.5,
        margin=dict(l=40, r=40, t=40, b=40)
    )

    # Chart 3: Daily Number of Transactions
    if 'Purchase_Date' in filtered_df.columns:
        daily_transactions = filtered_df.groupby('Purchase_Date', observed=False).size().reset_index(name='Number of Transactions')
    else:
        daily_transactions = pd.DataFrame({'Purchase_Date': [], 'Number of Transactions': []})
        print("Warning: Purchase_Date not in filtered_df; using fallback data.")
    fig3 = px.line(
        daily_transactions,
        x='Purchase_Date',
        y='Number of Transactions',
        title='Daily Number of Transactions',
        template='plotly_white'
    )
    fig3.update_layout(
        xaxis_title='Date',
        yaxis_title='Number of Transactions',
        xaxis_tickformat='%b %d',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
        font=dict(size=12),
        title_x=0.5,
        margin=dict(l=40, r=40, t=40, b=40)
    )

    # Chart 4: Journey Status Distribution
    if 'Journey_Status' in filtered_df.columns:
        journey_status_dist = filtered_df['Journey_Status'].value_counts().reset_index()
        journey_status_dist.columns = ['Journey_Status', 'Count']
    else:
        journey_status_dist = pd.DataFrame({'Journey_Status': [], 'Count': []})
        print("Warning: Journey_Status not in filtered_df.")
    fig4 = px.pie(
        journey_status_dist,
        names='Journey_Status',
        values='Count',
        title='Journey Status Distribution',
        hole=0.5,
        template='plotly_white'
    )
    fig4.update_traces(textinfo='percent+label', pull=[0.05]*len(journey_status_dist))
    fig4.update_layout(
        showlegend=True,
        font=dict(size=12),
        title_x=0.5,
        margin=dict(l=10, r=10, t=80, b=10)
    )

    return fig1, fig2, fig3, fig4

# Update Revenue Charts
@app.callback(
    [
        Output('chart-daily-revenue', 'figure'),
        Output('chart-ticket-class-revenue', 'figure'),
        Output('chart-station-revenue', 'figure')
    ],
    [
        Input('filter-month', 'value'),
        Input('filter-station', 'value'),
        Input('filter-ticket-type', 'value'),
        Input('filter-railcard', 'value'),
        Input('filter-payment', 'value')
    ]
)
def update_revenue_charts(month, station, ticket_type, railcard, payment):
    # Filter dataframe
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

    # Debug: Print filtered dataframe info
    print(f"Revenue charts - Filtered dataframe shape: {filtered_df.shape}")
    print(f"Revenue charts - Filtered dataframe columns: {filtered_df.columns.tolist()}")
    if not filtered_df.empty:
        if 'Journey_Date' in filtered_df.columns:
            print(f"Revenue charts - Sample Journey_Date: {filtered_df['Journey_Date'].head().tolist()}")
        if 'Ticket_Class' in filtered_df.columns:
            print(f"Revenue charts - Sample Ticket_Class: {filtered_df['Ticket_Class'].head().tolist()}")
        if 'Departure_Station_Name' in filtered_df.columns:
            print(f"Revenue charts - Sample Departure_Station_Name: {filtered_df['Departure_Station_Name'].head().tolist()}")

    # Handle empty dataframe
    if filtered_df.empty:
        empty_fig = px.scatter(x=[0], y=[0], title="No Data Available").update_traces(visible=False)
        return empty_fig, empty_fig, empty_fig

    # Chart 1: Daily Revenue
    if 'Journey_Date' in filtered_df.columns and 'Price' in filtered_df.columns:
        daily_revenue = filtered_df.groupby('Journey_Date', observed=False)['Price'].sum().reset_index(name='Daily Revenue')
    else:
        daily_revenue = pd.DataFrame({'Journey_Date': [], 'Daily Revenue': []})
        print("Warning: Journey_Date or Price not in filtered_df; using fallback data.")
    fig1 = px.line(
        daily_revenue,
        x='Journey_Date',
        y='Daily Revenue',
        title='Daily Revenue',
        template='plotly_white'
    )
    fig1.update_layout(
        xaxis_title='Date',
        yaxis_title='Revenue ($)',
        xaxis_tickformat='%b %d',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
        font=dict(size=12),
        title_x=0.5,
        margin=dict(l=40, r=40, t=40, b=40),
        showlegend=False
    )

    # Chart 2: Revenue Distribution by Ticket Class
    if 'Ticket_Class' in filtered_df.columns and 'Price' in filtered_df.columns:
        ticket_class_revenue = filtered_df.groupby('Ticket_Class', observed=False)['Price'].sum().reset_index()
    else:
        ticket_class_revenue = pd.DataFrame({'Ticket_Class': [], 'Price': []})
        print("Warning: Ticket_Class or Price not in filtered_df.")
    fig2 = px.pie(
        ticket_class_revenue,
        names='Ticket_Class',
        values='Price',
        title='Revenue Distribution by Ticket Class',
        template='plotly_white'
    )
    fig2.update_traces(textinfo='percent+label')
    fig2.update_layout(
        showlegend=True,
        font=dict(size=12),
        title_x=0.5,
        margin=dict(l=40, r=40, t=40, b=40)
    )

    # Chart 3: Revenue by Departure Station (Top 5)
    if 'Departure_Station_Name' in filtered_df.columns and 'Price' in filtered_df.columns:
        station_revenue = filtered_df.groupby('Departure_Station_Name', observed=False)['Price'].sum().sort_values(ascending=False).head(5).reset_index()
    else:
        station_revenue = pd.DataFrame({'Departure_Station_Name': [], 'Price': []})
        print("Warning: Departure_Station_Name or Price not in filtered_df.")
    fig3 = px.bar(
        station_revenue,
        x='Price',
        y='Departure_Station_Name',
        title='Revenue by Departure Station',
        template='plotly_white'
    )
    fig3.update_layout(
        xaxis_title='Revenue',
        yaxis_title='Station',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
        font=dict(size=12),
        title_x=0.5,
        margin=dict(l=40, r=40, t=40, b=40)
    )

    return fig1, fig2, fig3

# Update Journey Charts
@app.callback(
    [
        Output('chart-delay-reasons', 'figure'),
        Output('chart-railcard-usage', 'figure'),
        Output('chart-avg-price-ticket', 'figure'),
        Output('chart-purchase-type', 'figure')
    ],
    [
        Input('filter-month', 'value'),
        Input('filter-station', 'value'),
        Input('filter-ticket-type', 'value'),
        Input('filter-railcard', 'value'),
        Input('filter-payment', 'value')
    ]
)
def update_journey_charts(month, station, ticket_type, railcard, payment):
    # Filter dataframe
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

    # Debug: Print filtered dataframe info
    print(f"Journey charts - Filtered dataframe shape: {filtered_df.shape}")
    print(f"Journey charts - Filtered dataframe columns: {filtered_df.columns.tolist()}")
    if not filtered_df.empty:
        if 'Reason_for_Delay' in filtered_df.columns:
            print(f"Journey charts - Sample Reason_for_Delay: {filtered_df['Reason_for_Delay'].head().tolist()}")
        if 'Railcard' in filtered_df.columns:
            print(f"Journey charts - Sample Railcard: {filtered_df['Railcard'].head().tolist()}")
        if 'Ticket_Type' in filtered_df.columns:
            print(f"Journey charts - Sample Ticket_Type: {filtered_df['Ticket_Type'].head().tolist()}")
        if 'Purchase_Type' in filtered_df.columns:
            print(f"Journey charts - Sample Purchase_Type: {filtered_df['Purchase_Type'].head().tolist()}")

    # Handle empty dataframe
    if filtered_df.empty:
        empty_fig = px.scatter(x=[0], y=[0], title="No Data Available").update_traces(visible=False)
        return empty_fig, empty_fig, empty_fig, empty_fig

    # Chart 1: Delay Reasons (excluding 'No Delay')
    if 'Reason_for_Delay' in filtered_df.columns:
        delay_reasons = filtered_df[filtered_df['Reason_for_Delay'] != 'No Delay']['Reason_for_Delay'].value_counts().reset_index()
        delay_reasons.columns = ['Reason', 'Count']
    else:
        delay_reasons = pd.DataFrame({'Reason': [], 'Count': []})
        print("Warning: Reason_for_Delay not in filtered_df.")
    fig1 = px.bar(
        delay_reasons,
        x='Count',
        y='Reason',
        title='Delay Reasons',
        template='plotly_white'
    )
    fig1.update_layout(
        xaxis_title='Count',
        yaxis_title='Reason',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
        font=dict(size=12),
        title_x=0.5,
        margin=dict(l=40, r=40, t=40, b=40)
    )

    # Chart 2: Railcard Usage
    if 'Railcard' in filtered_df.columns:
        railcard_usage = filtered_df['Railcard'].value_counts().reset_index()
        railcard_usage.columns = ['Railcard Type', 'Number of Transactions']
    else:
        railcard_usage = pd.DataFrame({'Railcard Type': [], 'Number of Transactions': []})
        print("Warning: Railcard not in filtered_df.")
    fig2 = px.bar(
        railcard_usage,
        x='Railcard Type',
        y='Number of Transactions',
        title='Railcard Usage',
        template='plotly_white'
    )
    fig2.update_layout(
        xaxis_title='Railcard Type',
        yaxis_title='Number of Transactions',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
        font=dict(size=12),
        title_x=0.5,
        margin=dict(l=40, r=40, t=40, b=40)
    )

    # Chart 3: Average Price by Ticket Type
    if 'Ticket_Type' in filtered_df.columns and 'Price' in filtered_df.columns:
        avg_price_by_ticket = filtered_df.groupby('Ticket_Type', observed=False)['Price'].mean().reset_index()
    else:
        avg_price_by_ticket = pd.DataFrame({'Ticket_Type': [], 'Price': []})
        print("Warning: Ticket_Type or Price not in filtered_df.")
    fig3 = px.bar(
        avg_price_by_ticket,
        x='Ticket_Type',
        y='Price',
        title='Average Price by Ticket Type',
        template='plotly_white'
    )
    fig3.update_layout(
        xaxis_title='Ticket Type',
        yaxis_title='Average Price ($)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
        font=dict(size=12),
        title_x=0.5,
        margin=dict(l=40, r=40, t=40, b=40)
    )

    # Chart 4: Number of Transactions by Purchase Type
    if 'Purchase_Type' in filtered_df.columns:
        purchase_type_counts = filtered_df['Purchase_Type'].value_counts().reset_index()
        purchase_type_counts.columns = ['Purchase_Type', 'Count']
    else:
        purchase_type_counts = pd.DataFrame({'Purchase_Type': [], 'Count': []})
        print("Warning: Purchase_Type not in filtered_df.")
    fig4 = px.pie(
        purchase_type_counts,
        names='Purchase_Type',
        values='Count',
        title='Number of Transactions by Purchase Type',
        template='plotly_white'
    )
    fig4.update_traces(textinfo='percent+label')
    fig4.update_layout(
        showlegend=True,
        font=dict(size=12),
        title_x=0.5,
        margin=dict(l=40, r=40, t=40, b=40)
    )

    return fig1, fig2, fig3, fig4

# Update Performance Charts
# Update Performance Charts
@app.callback(
    [
        Output('chart-revenue-refunded', 'figure'),
        Output('chart-refunded-proportion', 'figure'),
        Output('chart-refunded-count', 'figure'),
        Output('chart-payment-method', 'figure')
    ],
    [
        Input('filter-month', 'value'),
        Input('filter-station', 'value'),
        Input('filter-ticket-type', 'value'),
        Input('filter-railcard', 'value'),
        Input('filter-payment', 'value')
    ]
)
def update_performance_charts(month, station, ticket_type, railcard, payment):
    # Filter dataframe
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

    # Handle empty dataframe
    if filtered_df.empty:
        empty_fig = px.scatter(x=[0], y=[0], title="No Data Available").update_traces(visible=False)
        return empty_fig, empty_fig, empty_fig, empty_fig

    # Chart 1: Revenue Impact of Refund Requests by Journey Status
    if 'Journey_Status' in filtered_df.columns and 'Refund_Request' in filtered_df.columns and 'Price' in filtered_df.columns:
        revenue_refunded = filtered_df.groupby(['Journey_Status', 'Refund_Request'], observed=False)['Price'].sum().reset_index()
        if not revenue_refunded.empty:
            fig1 = px.bar(
                revenue_refunded,
                x='Journey_Status',
                y='Price',
                color='Refund_Request',
                barmode='group',
                title='Revenue by Journey Status and Refund Request'
            )
            fig1.update_layout(
                xaxis_title='Journey Status',
                yaxis_title='Revenue ($)',
                plot_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
                font=dict(size=12),
                title_x=0.5,
                margin=dict(l=40, r=40, t=40, b=40),
                legend_title_text='Refund Requested'
            )
        else:
            fig1 = px.scatter(x=[0], y=[0], title="No Data Available").update_traces(visible=False)
    else:
        fig1 = px.scatter(x=[0], y=[0], title="No Data Available").update_traces(visible=False)

    # Chart 2: Proportion of Refund Requests
    if 'Refund_Request' in filtered_df.columns:
        refund_proportion = filtered_df['Refund_Request'].value_counts().reset_index()
        refund_proportion.columns = ['Refund_Request', 'Count']
        if not refund_proportion.empty:
            fig2 = px.pie(
                refund_proportion,
                names='Refund_Request',
                values='Count',
                title='Proportion of Refund Requests'
            )
            fig2.update_traces(textinfo='percent+label')
            fig2.update_layout(
                showlegend=True,
                font=dict(size=12),
                title_x=0.5,
                margin=dict(l=40, r=40, t=70, b=10)
            )
        else:
            fig2 = px.scatter(x=[0], y=[0], title="No Data Available").update_traces(visible=False)
    else:
        fig2 = px.scatter(x=[0], y=[0], title="No Data Available").update_traces(visible=False)

    # Chart 3: Refund Requests by Journey Status
    if 'Journey_Status' in filtered_df.columns and 'Refund_Request' in filtered_df.columns:
        # Group by Journey_Status and Refund_Request to include both 'Yes' and 'No'
        refund_count = filtered_df.groupby(['Journey_Status', 'Refund_Request'], observed=False).size().reset_index(name='Count')
        if not refund_count.empty:
            fig3 = px.bar(
                refund_count,
                x='Count',
                y='Journey_Status',
                color='Refund_Request',  # Differentiate 'Yes' and 'No' with colors
                barmode='group',         # Display bars side by side
                title='Refund Requests by Journey Status'
            )
            fig3.update_layout(
                xaxis_title='Number of Transactions',
                yaxis_title='Journey Status',
                plot_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
                font=dict(size=12),
                title_x=0.5,
                margin=dict(l=40, r=40, t=40, b=40),
                legend_title_text='Refund Requested'  # Clarify legend
            )
        else:
            fig3 = px.scatter(x=[0], y=[0], title="No Data Available").update_traces(visible=False)
    else:
        fig3 = px.scatter(x=[0], y=[0], title="No Data Available").update_traces(visible=False)

    # Chart 4: Payment Method Distribution
    if 'Payment_Method' in filtered_df.columns:
        payment_method_dist = filtered_df['Payment_Method'].value_counts().reset_index()
        payment_method_dist.columns = ['Payment_Method', 'Count']
        if not payment_method_dist.empty:
            fig4 = px.pie(
                payment_method_dist,
                names='Payment_Method',
                values='Count',
                title='Payment Method Distribution'
            )
            fig4.update_traces(textinfo='percent+label')
            fig4.update_layout(
                showlegend=True,
                font=dict(size=12),
                title_x=0.5,
                margin=dict(l=40, r=40, t=40, b=40)
            )
        else:
            fig4 = px.scatter(x=[0], y=[0], title="No Data Available").update_traces(visible=False)
    else:
        fig4 = px.scatter(x=[0], y=[0], title="No Data Available").update_traces(visible=False)

    return fig1, fig2, fig3, fig4

# Run App
if __name__ == '__main__':
    app.run(debug=True)
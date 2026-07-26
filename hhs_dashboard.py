import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import warnings
warnings.filterwarnings('ignore')


st.set_page_config(
    page_title="HHS Unaccompanied Alien Children Program Dashboard",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stMetric {
        background-color: #ffffff;
        padding: 0.5rem;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data(file_path):
    """Load and preprocess the HHS data"""
    try:
        df = pd.read_csv(file_path)
        
       
        df = df.dropna(how='all')
        
        
        df['Date'] = pd.to_datetime(df['Date'], format='%B %d, %Y')
        df = df.sort_values('Date', ascending=False)
        
       
        numeric_cols = ['Children apprehended and placed in CBP custody*', 
                       'Children in CBP custody', 
                       'Children transferred out of CBP custody',
                       'Children in HHS Care', 
                       'Children discharged from HHS Care']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '').str.strip()
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        
        df['Net_Intake'] = df['Children apprehended and placed in CBP custody*'] - df['Children transferred out of CBP custody']
        df['Total_In_Custody'] = df['Children in CBP custody'] + df['Children in HHS Care']
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()


FILE_PATH = "C:\Users\Lenovo\HHS_Unaccompanied_Alien_Children_Program (1).csv"

df = load_data(FILE_PATH)

if df.empty:
    st.error("Failed to load data. Please check the file path.")
    st.stop()


st.markdown('<h1 class="main-header">👶 HHS Unaccompanied Alien Children Program</h1>', unsafe_allow_html=True)
st.markdown("---")


st.sidebar.header("📊 Dashboard Controls")


min_date = df['Date'].min()
max_date = df['Date'].max()
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)


time_granularity = st.sidebar.selectbox(
    "Time Granularity",
    options=['Daily', 'Weekly', 'Monthly'],
    index=0
)


st.sidebar.header("📈 Metric Toggles")
show_apprehended = st.sidebar.checkbox("Children Apprehended", value=True)
show_cbp_custody = st.sidebar.checkbox("Children in CBP Custody", value=True)
show_hhs_care = st.sidebar.checkbox("Children in HHS Care", value=True)
show_discharged = st.sidebar.checkbox("Children Discharged", value=True)
show_total_custody = st.sidebar.checkbox("Total in Custody", value=True)


if len(date_range) == 2:
    start_date, end_date = date_range
    df_filtered = df[(df['Date'] >= pd.to_datetime(start_date)) & 
                     (df['Date'] <= pd.to_datetime(end_date))]
else:
    df_filtered = df.copy()


def resample_data(df, granularity):
    if granularity == 'Daily':
        return df
    elif granularity == 'Weekly':
        return df.resample('W-MON', on='Date').agg({
            'Children apprehended and placed in CBP custody*': 'sum',
            'Children in CBP custody': 'sum',
            'Children transferred out of CBP custody': 'sum',
            'Children in HHS Care': 'sum',
            'Children discharged from HHS Care': 'sum',
            'Net_Intake': 'sum',
            'Total_In_Custody': 'sum'
        }).reset_index()
    else:  
        return df.resample('M', on='Date').agg({
            'Children apprehended and placed in CBP custody*': 'sum',
            'Children in CBP custody': 'sum',
            'Children transferred out of CBP custody': 'sum',
            'Children in HHS Care': 'sum',
            'Children discharged from HHS Care': 'sum',
            'Net_Intake': 'sum',
            'Total_In_Custody': 'sum'
        }).reset_index()

df_resampled = resample_data(df_filtered, time_granularity)


st.header("📊 Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_apprehended = df_resampled['Children apprehended and placed in CBP custody*'].sum()
    st.metric(
        "Total Children Apprehended",
        f"{total_apprehended:,.0f}",
        delta="Last 30 days"
    )

with col2:
    avg_cbp_custody = df_resampled['Children in CBP custody'].mean()
    st.metric(
        "Avg CBP Custody",
        f"{avg_cbp_custody:,.0f}",
        delta="Current period"
    )

with col3:
    avg_hhs_care = df_resampled['Children in HHS Care'].mean()
    st.metric(
        "Avg HHS Care",
        f"{avg_hhs_care:,.0f}",
        delta="Current period"
    )

with col4:
    total_discharged = df_resampled['Children discharged from HHS Care'].sum()
    st.metric(
        "Total Children Discharged",
        f"{total_discharged:,.0f}",
        delta="Successfully placed"
    )

st.markdown("---")


st.header("📈 System Load Overview")


col1, col2 = st.columns(2)

with col1:
    
    fig_load = go.Figure()
    
    if show_apprehended:
        fig_load.add_trace(go.Scatter(
            x=df_resampled['Date'],
            y=df_resampled['Children apprehended and placed in CBP custody*'],
            name='Apprehended',
            line=dict(color='#e74c3c', width=2),
            fill='tozeroy',
            fillcolor='rgba(231, 76, 60, 0.2)'
        ))
    
    if show_cbp_custody:
        fig_load.add_trace(go.Scatter(
            x=df_resampled['Date'],
            y=df_resampled['Children in CBP custody'],
            name='CBP Custody',
            line=dict(color='#f39c12', width=2)
        ))
    
    if show_hhs_care:
        fig_load.add_trace(go.Scatter(
            x=df_resampled['Date'],
            y=df_resampled['Children in HHS Care'],
            name='HHS Care',
            line=dict(color='#2ecc71', width=2)
        ))
    
    if show_discharged:
        fig_load.add_trace(go.Scatter(
            x=df_resampled['Date'],
            y=df_resampled['Children discharged from HHS Care'],
            name='Discharged',
            line=dict(color='#3498db', width=2)
        ))
    
    fig_load.update_layout(
        title='Children Flow Overview',
        xaxis_title='Date',
        yaxis_title='Number of Children',
        height=400,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    st.plotly_chart(fig_load, use_container_width=True)

with col2:
    
    fig_bar = go.Figure()
    
    df_melted = df_resampled.melt(
        id_vars=['Date'],
        value_vars=['Children in CBP custody', 'Children in HHS Care'],
        var_name='Custody Type',
        value_name='Number'
    )
    
    fig_bar.add_trace(go.Bar(
        x=df_melted[df_melted['Custody Type'] == 'Children in CBP custody']['Date'],
        y=df_melted[df_melted['Custody Type'] == 'Children in CBP custody']['Number'],
        name='CBP Custody',
        marker_color='#f39c12'
    ))
    
    fig_bar.add_trace(go.Bar(
        x=df_melted[df_melted['Custody Type'] == 'Children in HHS Care']['Date'],
        y=df_melted[df_melted['Custody Type'] == 'Children in HHS Care']['Number'],
        name='HHS Care',
        marker_color='#2ecc71'
    ))
    
    fig_bar.update_layout(
        title='CBP vs HHS Custody Comparison',
        xaxis_title='Date',
        yaxis_title='Number of Children',
        height=400,
        barmode='stack',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")


st.header("🔄 CBP vs HHS Load Comparison")

col1, col2 = st.columns([2, 1])

with col1:
    
    fig_comparison = go.Figure()
    
    fig_comparison.add_trace(go.Scatter(
        x=df_resampled['Date'],
        y=df_resampled['Children in CBP custody'],
        name='CBP Custody',
        fill='tozeroy',
        line=dict(color='#f39c12', width=2),
        fillcolor='rgba(243, 156, 18, 0.3)'
    ))
    
    fig_comparison.add_trace(go.Scatter(
        x=df_resampled['Date'],
        y=df_resampled['Children in HHS Care'],
        name='HHS Care',
        fill='tonexty',
        line=dict(color='#2ecc71', width=2),
        fillcolor='rgba(46, 204, 113, 0.3)'
    ))
    
    fig_comparison.update_layout(
        title='CBP Custody vs HHS Care Over Time',
        xaxis_title='Date',
        yaxis_title='Number of Children',
        height=400,
        hovermode='x unified'
    )
    st.plotly_chart(fig_comparison, use_container_width=True)

with col2:
   
    st.subheader("Current Snapshot")
    
    latest_data = df_resampled.iloc[0]
    
    st.markdown(f"""
    <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 10px;">
        <p style="font-size: 1.2rem; margin: 0;">📊 <strong>CBP Custody</strong></p>
        <p style="font-size: 2rem; color: #f39c12; margin: 0;">{latest_data['Children in CBP custody']:,.0f}</p>
        <hr style="margin: 0.5rem 0;">
        <p style="font-size: 1.2rem; margin: 0;">🏠 <strong>HHS Care</strong></p>
        <p style="font-size: 2rem; color: #2ecc71; margin: 0;">{latest_data['Children in HHS Care']:,.0f}</p>
        <hr style="margin: 0.5rem 0;">
        <p style="font-size: 1.2rem; margin: 0;">📈 <strong>Total in Custody</strong></p>
        <p style="font-size: 2rem; color: #3498db; margin: 0;">{latest_data['Total_In_Custody']:,.0f}</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")


st.header("📊 Net Intake & Backlog Trends")

col1, col2 = st.columns(2)

with col1:
    
    fig_intake = go.Figure()
    
    fig_intake.add_trace(go.Scatter(
        x=df_resampled['Date'],
        y=df_resampled['Net_Intake'],
        mode='lines+markers',
        name='Net Intake',
        line=dict(color='#9b59b6', width=3),
        marker=dict(size=6)
    ))
    
   
    fig_intake.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)
    
    fig_intake.update_layout(
        title='Net Intake (Apprehended - Transferred Out)',
        xaxis_title='Date',
        yaxis_title='Net Intake',
        height=400,
        hovermode='x unified'
    )
    st.plotly_chart(fig_intake, use_container_width=True)

with col2:
    
    fig_backlog = go.Figure()
    
    fig_backlog.add_trace(go.Scatter(
        x=df_resampled['Date'],
        y=df_resampled['Children in HHS Care'],
        name='HHS Care Backlog',
        line=dict(color='#e67e22', width=3),
        fill='tozeroy',
        fillcolor='rgba(230, 126, 34, 0.2)'
    ))
    
    fig_backlog.add_trace(go.Scatter(
        x=df_resampled['Date'],
        y=df_resampled['Children in CBP custody'],
        name='CBP Custody Backlog',
        line=dict(color='#16a085', width=3),
        fill='tozeroy',
        fillcolor='rgba(22, 160, 133, 0.2)'
    ))
    
    fig_backlog.update_layout(
        title='Backlog Trends',
        xaxis_title='Date',
        yaxis_title='Number of Children',
        height=400,
        hovermode='x unified'
    )
    st.plotly_chart(fig_backlog, use_container_width=True)

st.markdown("---")


st.header("💡 Key Insights")

col1, col2, col3 = st.columns(3)

with col1:
    
    peak_cbp = df_resampled['Children in CBP custody'].max()
    peak_cbp_date = df_resampled[df_resampled['Children in CBP custody'] == peak_cbp]['Date'].iloc[0]
    
    st.metric(
        "Peak CBP Custody",
        f"{peak_cbp:,.0f}",
        delta=f"on {peak_cbp_date.strftime('%Y-%m-%d')}"
    )

with col2:
    peak_hhs = df_resampled['Children in HHS Care'].max()
    peak_hhs_date = df_resampled[df_resampled['Children in HHS Care'] == peak_hhs]['Date'].iloc[0]
    
    st.metric(
        "Peak HHS Care",
        f"{peak_hhs:,.0f}",
        delta=f"on {peak_hhs_date.strftime('%Y-%m-%d')}"
    )

with col3:
    
    total_children = df_resampled['Total_In_Custody'].sum()
    avg_custody = df_resampled['Total_In_Custody'].mean()
    
    st.metric(
        "Avg Total Custody",
        f"{avg_custody:,.0f}",
        delta="Overall average"
    )

st.markdown("---")


with st.expander("📋 View Raw Data"):
    st.dataframe(
        df_resampled,
        use_container_width=True,
        column_config={
            'Date': st.column_config.DateColumn('Date'),
            'Children apprehended and placed in CBP custody*': st.column_config.NumberColumn('Apprehended'),
            'Children in CBP custody': st.column_config.NumberColumn('CBP Custody'),
            'Children transferred out of CBP custody': st.column_config.NumberColumn('Transferred Out'),
            'Children in HHS Care': st.column_config.NumberColumn('HHS Care'),
            'Children discharged from HHS Care': st.column_config.NumberColumn('Discharged'),
            'Net_Intake': st.column_config.NumberColumn('Net Intake'),
            'Total_In_Custody': st.column_config.NumberColumn('Total Custody')
        }
    )

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

csv_data = convert_df_to_csv(df_resampled)
st.download_button(
    label="📥 Download Data as CSV",
    data=csv_data,
    file_name=f"hhs_data_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
    mime="text/csv",
)


st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>📊 HHS Unaccompanied Alien Children Program Dashboard</p>
        <p style="font-size: 0.8rem;">Data updated daily | Built with Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)

import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd
import requests
from bs4 import BeautifulSoup
from utils import get_market_data, load_historical_decisions
from agents.coordinator_agent import CoordinatorAgent
from config import AGENT_SETTINGS, TRADING_SETTINGS, NEWS_SOURCES
import os
import json

# Page config
st.set_page_config(
    page_title="Multi-Agent Quants - AI Trading Analysis",
    page_icon="utils/icon.ico",
    layout="wide"
)

st.markdown("""
    <style>
    /* Global styles */
    .stApp {
        background-color: #0E1117;
        color: white;
    }
    
    .big-font {
        font-size: 28px !important;
        font-weight: bold;
        color: white;
    }
    
    .analysis-box {
        background-color: #1E1E1E;
        padding: 25px;
        border-radius: 12px;
        margin: 15px 0;
        color: white;
        line-height: 1.6;
    }
    
    .signal-positive {
        color: #00FF00;
        font-weight: bold;
        background-color: rgba(0, 255, 0, 0.1);
        padding: 8px 12px;
        border-radius: 6px;
    }
    
    .signal-negative {
        color: #FF0000;
        font-weight: bold;
        background-color: rgba(255, 0, 0, 0.1);
        padding: 8px 12px;
        border-radius: 6px;
    }
    
    .analysis-box h3 {
        color: white;
        margin-bottom: 12px;
    }
    
    .analysis-box p {
        color: white;
    }
    
    /* Additional styles for Streamlit elements */
    .stMarkdown, .stText, .stButton, .stSelectbox {
        color: white;
    }
    
    .streamlit-expanderHeader {
        background-color: #1E1E1E !important;
        color: white !important;
    }
    
    div[data-testid="stDataFrame"] {
        background-color: #1E1E1E;
        color: white;
    }
    
    div[data-testid="stDataFrame"] td {
        background-color: #2C2C2C;
        color: white;
    }
    
    div[data-testid="stDataFrame"] th {
        background-color: #1E1E1E;
        color: white;
    }
    
    .stSidebar {
        background-color: #0E1117;
        color: white;
    }
    
    .stProgress > div > div {
        background-color: #1E1E1E;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_available_symbols():
    """Get list of available symbols from utils/symbols.txt file"""
    try:
        with open('utils/symbols.txt', 'r') as f:
            symbols = [line.strip() for line in f if line.strip()]
        
        return symbols

    except FileNotFoundError:
        st.error("Could not find utils/symbols.txt file")
        return []
    except Exception as e:
        st.error(f"Error loading symbols: {str(e)}")
        return []

def plot_stock_price(symbol: str):
    """Create a stock price chart using plotly"""
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period='6mo')
    
    fig = go.Figure(data=[go.Candlestick(x=hist.index,
                open=hist['Open'],
                high=hist['High'],
                low=hist['Low'],
                close=hist['Close'])])
    
    fig.update_layout(
        title=f'{symbol} Stock Price',
        yaxis_title='Price',
        xaxis_title='Date',
        height=400
    )
    
    return fig

def display_analysis_results(result):
    """Display the analysis results in a structured format"""
    # Display confidence score
    st.markdown('### Analysis Confidence')
    st.progress(result['confidence_score'])
    st.write(f"{result['confidence_score']:.2%} confidence in analysis")

    # Display symbol signals
    signals = result['final_decision']['symbol_signals']
    print(signals)
    cols = st.columns(len(signals))
    
    for col, (symbol, signal) in zip(cols, signals.items()):
        with col:
            st.markdown(
                f"<div class='analysis-box'>"
                f"<h3>{symbol}</h3>"
                f"<p class='{'signal-positive' if signal else 'signal-negative'}'>"
                f"{'BULLISH' if signal else 'BEARISH'}</p>"
                f"</div>",
                unsafe_allow_html=True
            )
    
    # Display final decision
    with st.expander("View Detailed Analysis", expanded=True):
        st.markdown(f"<div class='analysis-box'>{result['final_decision']['decision']}</div>", 
                   unsafe_allow_html=True)
    
    # Display market context
    st.markdown('### Market Context')
    market_data = result['final_decision']['market_context']
    if market_data:
        df = pd.DataFrame(market_data).T
        
        # Style the dataframe
        styled_df = df.style.set_properties(**{
            'background-color': '#1E1E1E',
            'color': 'white',
            'border-color': '#2C2C2C',
            'font-size': '1rem',
            'padding': '0.5rem'
        }).format(precision=2)  # Round numeric values to 2 decimal places
        
        # Display the styled dataframe with custom parameters
        st.dataframe(
            styled_df,
            use_container_width=True,  # Make table full width
            height=min(len(market_data) * 35 + 38, 500),  # Dynamic height based on rows
        )
    else:
        st.info("No market context data available")

def update_config():
    """Update the configuration based on user inputs"""
    # Update agent settings
    for agent in AGENT_SETTINGS:
        AGENT_SETTINGS[agent]["model"] = st.session_state[f"{agent}_model"]
        AGENT_SETTINGS[agent]["temperature"] = st.session_state[f"{agent}_temp"]
        AGENT_SETTINGS[agent]["max_tokens"] = st.session_state[f"{agent}_tokens"]
    
    # Update trading settings
    TRADING_SETTINGS["analysis_timeframe"] = st.session_state.analysis_timeframe
    TRADING_SETTINGS["risk_tolerance"] = st.session_state.risk_tolerance

def show_agent_output(agent_type: str, output: dict):
    """Display the output of an individual agent"""
    with st.expander(f"{agent_type.replace('_', ' ').title()} Output", expanded=False):
        # st.markdown(f"<div class='analysis-box'>", unsafe_allow_html=True)
        
        # Display timestamp
        st.write(f"Analysis Time: {output.get('timestamp', 'N/A')}")
        
        # Display agent-specific outputs
        if agent_type == "news_agent":
            st.write("News Analysis:")
            st.markdown(output.get('news_analysis', 'No analysis available'))
            st.write("Analyzed Symbols:", ", ".join(output.get('analyzed_symbols', [])))
            
        elif agent_type == "reflection_agent":
            st.write("Reflection Analysis:")
            st.markdown(output.get('reflection_analysis', 'No analysis available'))
            if 'patterns_identified' in output:
                st.write("Patterns Identified:")
                st.json(output['patterns_identified'])
                
        elif agent_type == "debate_agent":
            st.write("Debate Analysis:")
            st.markdown(output.get('debate_analysis', 'No analysis available'))
            if 'debate_rounds' in output:
                for round_data in output['debate_rounds']:
                    st.write(f"\nRound {round_data['round']} ({round_data['perspective'].upper()}):")
                    st.markdown(round_data['arguments'])
            st.write(f"Confidence Score: {output.get('confidence_score', 0):.2%}")
        
        st.markdown("</div>", unsafe_allow_html=True)

def config_sidebar():
    """Create the configuration sidebar"""
    st.sidebar.title("Configuration")
    
    # API Key Configuration
    st.sidebar.subheader("OpenAI API Key")
    api_key = st.sidebar.text_input(
        "Enter your OpenAI API key",
        type="password",
        help="Your API key will not be stored and will only be used for this session",
        key="openai_api_key"
    )
    
    # Agent Enable/Disable Section
    st.sidebar.subheader("Enable/Disable Agents")
    
    # Initialize session state for agent toggles if not exists
    if "enabled_agents" not in st.session_state:
        st.session_state.enabled_agents = {
            "news_agent": True,
            "reflection_agent": True,
            "debate_agent": True
        }
    
    # Create toggles for each agent except coordinator
    st.session_state.enabled_agents["news_agent"] = st.sidebar.checkbox(
        "News Agent",
        value=st.session_state.enabled_agents["news_agent"],
        help="Analyzes current news and market sentiment"
    )

    st.session_state.enabled_agents["reflection_agent"] = st.sidebar.checkbox(
        "Reflection Agent",
        value=st.session_state.enabled_agents["reflection_agent"],
        help="Analyzes historical decisions and patterns"
    )

    st.session_state.enabled_agents["debate_agent"] = st.sidebar.checkbox(
        "Debate Agent",
        value=st.session_state.enabled_agents["debate_agent"],
        help="Creates pros and cons analysis"
    )
    
    # Model Configuration Section
    st.sidebar.subheader("Model Settings")
    
    # Create tabs for different agent configurations
    agent_tabs = st.sidebar.tabs(["News", "Reflection", "Debate", "Coordinator"])
    
    for agent, tab in zip(AGENT_SETTINGS.keys(), agent_tabs):
        with tab:
            st.selectbox(
                "Model",
                options=["gpt-3.5-turbo", "gpt-4", "gpt-4o-mini", "gpt-4o"],
                key=f"{agent}_model",
                index=["gpt-3.5-turbo", "gpt-4", "gpt-4o-mini", "gpt-4o"].index(AGENT_SETTINGS[agent]["model"]) if AGENT_SETTINGS[agent]["model"] in ["gpt-3.5-turbo", "gpt-4", "gpt-4o-mini", "gpt-4o"] else 2
            )
            
            st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=AGENT_SETTINGS[agent]["temperature"],
                step=0.1,
                key=f"{agent}_temp"
            )
            
            st.number_input(
                "Max Tokens",
                min_value=100,
                max_value=4000,
                value=AGENT_SETTINGS[agent]["max_tokens"],
                step=100,
                key=f"{agent}_tokens"
            )
    
    # Trading Settings Section
    st.sidebar.subheader("Trading Settings")
    
    st.sidebar.selectbox(
        "Analysis Timeframe",
        options=["1d", "5d", "1mo", "3mo", "6mo", "1y"],
        index=["1d", "5d", "1mo", "3mo", "6mo", "1y"].index(TRADING_SETTINGS["analysis_timeframe"]),
        key="analysis_timeframe"
    )
    
    st.sidebar.slider(
        "Risk Tolerance",
        min_value=0.01,
        max_value=0.10,
        value=TRADING_SETTINGS["risk_tolerance"],
        step=0.01,
        format="%.2f",
        key="risk_tolerance",
        help="Maximum risk per trade (as a decimal)"
    )
    
    # News Sources Configuration
    st.sidebar.subheader("News Sources")
    news_sources = st.sidebar.multiselect(
        "Select News Sources",
        options=["reuters.com", "bloomberg.com", "wsj.com", "ft.com", "cnbc.com", "marketwatch.com"],
        default=NEWS_SOURCES
    )
    
    # Apply Configuration Button
    if st.sidebar.button("Apply Configuration"):
        update_config()
        st.sidebar.success("Configuration updated successfully!")

def main():
    st.title("🤖 Multi-Agent Quants - AI Trading Analysis")
    
    # Get configuration from sidebar
    config_sidebar()
    
    # Check for API key
    if not st.session_state.get("openai_api_key"):
        st.error("Please enter your OpenAI API key in the sidebar.")
        st.stop()
    
    # Set the API key for this session
    os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key
    
    # Symbol selection in main page
    st.markdown('<p class="big-font">Select Symbols to Analyze</p>', unsafe_allow_html=True)
    selected_symbols = st.multiselect(
        "Enter Stock Symbols to Analyze",
        options=get_available_symbols(),
        default=["AAPL", "MSFT", "GOOGL"],
        max_selections=5,
        help="Type to search for any stock symbol"
    )
    
    if not selected_symbols:
        st.warning("Please select at least one symbol to analyze.")
        st.stop()
    
    # Display stock charts
    st.markdown('<p class="big-font">Price Charts</p>', unsafe_allow_html=True)
    cols = st.columns(len(selected_symbols))
    for col, symbol in zip(cols, selected_symbols):
        with col:
            fig = plot_stock_price(symbol)
            st.plotly_chart(fig, use_container_width=True)
    
    # Create placeholder for real-time updates
    agent_outputs = st.empty()
    
    # Analysis button
    if st.button("Run Analysis", type="primary"):
        with st.spinner("Running AI analysis..."):
            # Initialize coordinator
            coordinator = CoordinatorAgent(AGENT_SETTINGS["coordinator_agent"])
            
            # Prepare data
            market_data = get_market_data(selected_symbols)
            historical_decisions = load_historical_decisions()
            
            # Create analysis context
            analysis_context = {
                "symbols": selected_symbols,
                "market_data": market_data,
                "historical_decisions": historical_decisions,
                "timestamp": datetime.now().isoformat(),
                "proposed_action": {
                    "type": "ANALYSIS",
                    "symbols": selected_symbols,
                    "risk_level": TRADING_SETTINGS["risk_tolerance"]
                },
                "enabled_agents": st.session_state.enabled_agents  # Pass enabled agents configuration
            }
            
            # Get trading decision
            try:
                # Initialize container for agent outputs
                with agent_outputs.container():
                    st.markdown('<p class="big-font">Agent Outputs</p>', unsafe_allow_html=True)
                    
                    # Only show enabled agent outputs
                    print(st.session_state.enabled_agents)
                    if st.session_state.enabled_agents["news_agent"]:
                        news_analysis = coordinator.news_agent.analyze({
                            "symbols": selected_symbols,
                            "timestamp": datetime.now().isoformat()
                        })
                        show_agent_output("news_agent", news_analysis)
                    
                    if st.session_state.enabled_agents["reflection_agent"]:
                        reflection_analysis = coordinator.reflection_agent.analyze({
                            "historical_decisions": historical_decisions,
                            "current_market": market_data,
                            "timestamp": datetime.now().isoformat()
                        })
                        show_agent_output("reflection_agent", reflection_analysis)
                    
                    if st.session_state.enabled_agents["debate_agent"]:
                        debate_analysis = coordinator.debate_agent.analyze({
                            "market_data": market_data,
                            "proposed_action": analysis_context["proposed_action"],
                            "timestamp": datetime.now().isoformat()
                        })
                        show_agent_output("debate_agent", debate_analysis)
                    
                    # Get final decision
                    result = coordinator._synthesize_analyses(
                        news_analysis if st.session_state.enabled_agents["news_agent"] else None,
                        reflection_analysis if st.session_state.enabled_agents["reflection_agent"] else None,
                        debate_analysis if st.session_state.enabled_agents["debate_agent"] else None,
                        analysis_context
                    )
                    
                    # Display final results
                    st.markdown('<p class="big-font">Final Analysis</p>', unsafe_allow_html=True)
                    display_analysis_results({
                        "confidence_score": coordinator._calculate_overall_confidence(
                            news_analysis if st.session_state.enabled_agents["news_agent"] else None,
                            reflection_analysis if st.session_state.enabled_agents["reflection_agent"] else None,
                            debate_analysis if st.session_state.enabled_agents["debate_agent"] else None,
                            st.session_state.enabled_agents
                        ),
                        "final_decision": result
                    })
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main() 
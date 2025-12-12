import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from networkx.algorithms.community import greedy_modularity_communities
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Wiki-Vote Analytics | Social Network Analysis",
    layout="wide",
    page_icon="🗳️",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS for a modern, professional look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .big-font { 
        font-size: 32px !important; 
        font-weight: 700; 
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .metric-card { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px; 
        border-radius: 15px; 
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        color: white;
    }
    
    .stTabs [data-baseweb="tab-list"] { 
        gap: 8px; 
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] { 
        height: 55px; 
        background: white;
        border-radius: 8px; 
        padding: 12px 20px;
        font-weight: 600;
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        border-color: #667eea;
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"] { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
    }
    
    .insight-box {
        background-color: #f0f7ff;
        border-left: 5px solid #667eea;
        padding: 20px;
        border-radius: 8px;
        margin: 20px 0;
        color: #1a237e !important;
    }
    
    .insight-box strong, .insight-box h4, .insight-box p, .insight-box ul, .insight-box li {
        color: #1a237e !important;
    }
    
    .warning-box {
        background-color: #fff4e5;
        border-left: 5px solid #ff9800;
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
        color: #e65100 !important;
    }
    
    .warning-box strong, .warning-box h4, .warning-box p, .warning-box ul, .warning-box li {
        color: #e65100 !important;
    }
    
    .success-box {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        padding: 20px;
        border-radius: 8px;
        margin: 15px 0;
        color: #1b5e20 !important;
    }
    
    .success-box strong, .success-box h4, .success-box p, .success-box ul, .success-box li {
        color: #1b5e20 !important;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
        color: #667eea;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] .element-container {
        color: white !important;
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    [data-testid="stSidebar"] label {
        color: white !important;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 12px 28px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Ensure all markdown text is readable */
    .stMarkdown {
        color: #333333;
    }
    
    /* Fix for any text in colored backgrounds */
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] li,
    div[data-testid="stMarkdownContainer"] strong {
        color: inherit !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA LOADING (Cached)
# ==========================================
@st.cache_data
def load_data():
    try:
        # Load dataset - same as notebook
        G = nx.read_edgelist("Wiki-Vote.txt", create_using=nx.DiGraph(), nodetype=int)
        return G
    except FileNotFoundError:
        return None

G = load_data()

# Enhanced Sidebar Navigation
with st.sidebar:
    st.markdown("<h1 style='color: white; text-align: center;'>🗳️ Wiki-Vote Analytics</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #e0e0e0; text-align: center; font-size: 14px;'>Social Network Analysis Platform</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    page = st.radio("📍 Navigation", 
        ["🏠 Overview", 
         "📊 Node Degree Analysis",
         "📈 Network Statistics", 
         "👑 Power & Roles (Centrality)", 
         "🎨 Visualizations (Graphs)",
         "🌐 Community Detection"],
        label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("""
    <div style='background-color: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; color: white;'>
        <p style='margin: 0; font-weight: 600;'>📁 Dataset</p>
        <p style='margin: 5px 0 0 0; font-size: 13px;'>Wikipedia Admin Elections</p>
        <p style='margin: 5px 0 0 0; font-size: 12px; opacity: 0.8;'>7,115 nodes • 103,689 edges</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("<p style='color: #e0e0e0; text-align: center; font-size: 11px;'>Made with ❤️ using Streamlit</p>", unsafe_allow_html=True)

# ==========================================
# PAGE 1: HOME (Clean, No Visualizations)
# ==========================================
if page == "🏠 Overview":
    st.markdown("<div class='big-font'>🗳️ Wikipedia Voting Network Analysis</div>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #666; font-weight: 400;'>Understanding Power, Trust, and Community in Digital Democracy</h3>", unsafe_allow_html=True)
    
    if G is None:
        st.error("❌ Error: 'Wiki-Vote.txt' not found. Please place the file in the folder.")
        st.stop()

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Key Metrics Row with Enhanced Design
    st.markdown("#### 🔢 Network Overview")
    c1, c2, c3, c4 = st.columns(4)
    
    c1.metric("👥 Total Users", f"{G.number_of_nodes():,}", help="Total number of Wikipedia users in the network")
    c2.metric("🗳️ Total Votes", f"{G.number_of_edges():,}", help="Total voting interactions")
    c3.metric("🔗 Network Density", f"{nx.density(G):.5f}", help="How interconnected the network is (0=sparse, 1=complete)")
    
    # Calculate simple isolated stats
    in_degrees = [d for n, d in G.in_degree()]
    zeros = in_degrees.count(0)
    c4.metric("🤫 Silent Voters", f"{zeros:,}", delta=f"{zeros/len(in_degrees)*100:.1f}%", delta_color="off", help="Users who received no votes")

    # Interactive Quick Stats
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### ⚡ Quick Statistics")
    
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        avg_degree = G.number_of_edges() / G.number_of_nodes()
        max_degree = max(dict(G.degree()).values())
        st.markdown(f"""
        <div class='insight-box'>
            <h4>📈 Degree Insights</h4>
            <p><strong>Average Degree:</strong> {avg_degree:.2f} connections</p>
            <p><strong>Max Degree:</strong> {max_degree:,} connections</p>
            <p><em>The network shows high variance - a few super-connected nodes!</em></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_b:
        reciprocity = nx.reciprocity(G)
        st.markdown(f"""
        <div class='insight-box'>
            <h4>🤝 Reciprocity</h4>
            <p><strong>Mutual Votes:</strong> {reciprocity*100:.2f}%</p>
            <p><em>Low reciprocity indicates hierarchical voting patterns - voters support admins, but admins don't vote back equally.</em></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_c:
        # Get largest component size
        if G.is_directed():
            largest_wcc = len(max(nx.weakly_connected_components(G), key=len))
        else:
            largest_wcc = len(max(nx.connected_components(G), key=len))
        connectivity_pct = (largest_wcc / G.number_of_nodes()) * 100
        
        st.markdown(f"""
        <div class='insight-box'>
            <h4>🌐 Connectivity</h4>
            <p><strong>Giant Component:</strong> {largest_wcc:,} nodes ({connectivity_pct:.1f}%)</p>
            <p><em>Most users belong to one large interconnected community.</em></p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    #### 📖 Analysis Objectives
    
    <div style='background-color: #f8f9fa; padding: 25px; border-radius: 10px; margin-top: 20px;'>
    
    **🎯 What We'll Discover:**
    
    1. **👑 Power Analysis** - Identify the 'Kings', 'Brokers', and 'Influencers' 
    2. **📊 Network Metrics** - Measure connectivity, clustering, and small-world properties
    3. **🎨 Visual Insights** - See the network structure, communities, and voting patterns
    4. **🌐 Community Detection** - Find natural groupings and sub-communities
    5. **📈 Degree Distribution** - Understand the power-law nature of social networks
    
    </div>
    
    <br>
    
    <div class='success-box'>
        <strong>💡 Getting Started:</strong> Use the sidebar navigation to explore different aspects of the network. 
        Start with <strong>Node Degree Analysis</strong> for a deep dive into user connections!
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# PAGE 2: NODE DEGREE ANALYSIS
# ==========================================
elif page == "📊 Node Degree Analysis":
    st.markdown("<div class='big-font'>📊 Node Degree Analysis</div>", unsafe_allow_html=True)
    st.markdown("<p style='color: #666; font-size: 18px;'>Understanding Connections: How many links does each user have?</p>", unsafe_allow_html=True)
    
    # Real-world analogy
    st.markdown("""
    <div class='insight-box'>
        <h4>🌟 Real-World Analogy: Instagram</h4>
        <ul>
            <li><strong>In-Degree:</strong> How many followers do you have? (People who voted FOR you)</li>
            <li><strong>Out-Degree:</strong> How many people do you follow? (People you voted FOR)</li>
        </ul>
        <p><strong>In Our Network:</strong></p>
        <ul>
            <li>High In-Degree = You're <strong>popular/trusted</strong> 👑</li>
            <li>High Out-Degree = You're <strong>active in voting</strong> 🗳️</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Calculate degrees
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    
    # Basic Statistics
    st.markdown("### 📈 Degree Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    avg_in = np.mean(list(in_degrees.values()))
    avg_out = np.mean(list(out_degrees.values()))
    max_in = max(in_degrees.values())
    max_out = max(out_degrees.values())
    
    col1.metric("📥 Avg In-Degree", f"{avg_in:.2f}", help="Average votes received per user")
    col2.metric("📤 Avg Out-Degree", f"{avg_out:.2f}", help="Average votes cast per user")
    col3.metric("🏆 Max In-Degree", f"{max_in:,}", help="Most votes received by any user")
    col4.metric("⚡ Max Out-Degree", f"{max_out:,}", help="Most votes cast by any user")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Top Users
    tab1, tab2, tab3 = st.tabs(["👑 Most Popular Users", "🗳️ Most Active Voters", "📊 Degree Distribution"])
    
    with tab1:
        st.markdown("#### Top 15 Most Voted Users (Highest In-Degree)")
        st.caption("These users are the most trusted and popular in the network")
        
        top_in = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:15]
        df_top_in = pd.DataFrame(top_in, columns=['User ID', 'Votes Received'])
        df_top_in['Rank'] = range(1, len(df_top_in) + 1)
        df_top_in = df_top_in[['Rank', 'User ID', 'Votes Received']]
        
        # Create interactive bar chart
        fig_in = px.bar(df_top_in, x='User ID', y='Votes Received', 
                        title='Top 15 Users by In-Degree',
                        labels={'Votes Received': 'Number of Votes Received', 'User ID': 'User ID'},
                        color='Votes Received',
                        color_continuous_scale='Viridis',
                        text='Votes Received')
        fig_in.update_traces(texttemplate='%{text}', textposition='outside')
        fig_in.update_layout(showlegend=False, height=500)
        st.plotly_chart(fig_in, use_container_width=True)
        
        st.dataframe(df_top_in, use_container_width=True, hide_index=True)
    
    with tab2:
        st.markdown("#### Top 15 Most Active Voters (Highest Out-Degree)")
        st.caption("These users are the most engaged, casting the most votes")
        
        top_out = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)[:15]
        df_top_out = pd.DataFrame(top_out, columns=['User ID', 'Votes Cast'])
        df_top_out['Rank'] = range(1, len(df_top_out) + 1)
        df_top_out = df_top_out[['Rank', 'User ID', 'Votes Cast']]
        
        # Create interactive bar chart
        fig_out = px.bar(df_top_out, x='User ID', y='Votes Cast',
                         title='Top 15 Users by Out-Degree',
                         labels={'Votes Cast': 'Number of Votes Cast', 'User ID': 'User ID'},
                         color='Votes Cast',
                         color_continuous_scale='Plasma',
                         text='Votes Cast')
        fig_out.update_traces(texttemplate='%{text}', textposition='outside')
        fig_out.update_layout(showlegend=False, height=500)
        st.plotly_chart(fig_out, use_container_width=True)
        
        st.dataframe(df_top_out, use_container_width=True, hide_index=True)
    
    with tab3:
        st.markdown("#### 📊 Degree Distribution (Power-Law Pattern)")
        st.caption("Visualizing the 'rich-get-richer' phenomenon in social networks")
        
        # Get degree sequences
        in_degree_sequence = list(in_degrees.values())
        out_degree_sequence = list(out_degrees.values())
        
        # Create distribution plots
        fig = plt.figure(figsize=(16, 10))
        
        # In-Degree Linear
        ax1 = plt.subplot(2, 3, 1)
        in_counts = Counter(in_degree_sequence)
        degrees = sorted(in_counts.keys())
        counts = [in_counts[d] for d in degrees]
        ax1.bar(degrees[:50], counts[:50], color='#667eea', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('In-Degree (Votes Received)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Number of Users', fontsize=11, fontweight='bold')
        ax1.set_title('In-Degree Distribution (Linear)', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # In-Degree Log-Log
        ax2 = plt.subplot(2, 3, 2)
        ax2.loglog(degrees, counts, 'o', color='#667eea', alpha=0.6, markersize=6)
        ax2.set_xlabel('In-Degree [log]', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Frequency [log]', fontsize=11, fontweight='bold')
        ax2.set_title('In-Degree Distribution (Log-Log - Power Law)', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Out-Degree Linear
        ax3 = plt.subplot(2, 3, 4)
        out_counts = Counter(out_degree_sequence)
        out_degrees_sorted = sorted(out_counts.keys())
        out_counts_sorted = [out_counts[d] for d in out_degrees_sorted]
        ax3.bar(out_degrees_sorted[:50], out_counts_sorted[:50], color='#f093fb', alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Out-Degree (Votes Cast)', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Number of Users', fontsize=11, fontweight='bold')
        ax3.set_title('Out-Degree Distribution (Linear)', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Out-Degree Log-Log
        ax4 = plt.subplot(2, 3, 5)
        ax4.loglog(out_degrees_sorted, out_counts_sorted, 'o', color='#f093fb', alpha=0.6, markersize=6)
        ax4.set_xlabel('Out-Degree [log]', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Frequency [log]', fontsize=11, fontweight='bold')
        ax4.set_title('Out-Degree Distribution (Log-Log - Power Law)', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Combined comparison
        ax5 = plt.subplot(2, 3, 3)
        ax5.hist([in_degree_sequence, out_degree_sequence], bins=50, label=['In-Degree', 'Out-Degree'],
                 color=['#667eea', '#f093fb'], alpha=0.6, edgecolor='black')
        ax5.set_xlabel('Degree', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax5.set_title('In vs Out Degree Comparison', fontsize=13, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Statistical summary box
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        summary_text = f"""
        STATISTICAL SUMMARY
        
        In-Degree:
        • Mean: {np.mean(in_degree_sequence):.2f}
        • Median: {np.median(in_degree_sequence):.2f}
        • Std Dev: {np.std(in_degree_sequence):.2f}
        • Max: {max(in_degree_sequence)}
        
        Out-Degree:
        • Mean: {np.mean(out_degree_sequence):.2f}
        • Median: {np.median(out_degree_sequence):.2f}
        • Std Dev: {np.std(out_degree_sequence):.2f}
        • Max: {max(out_degree_sequence)}
        """
        ax6.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                verticalalignment='center')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
        <div class='success-box'>
            <h4>🔬 Power-Law Interpretation</h4>
            <p>The log-log plots show approximately <strong>linear patterns</strong>, confirming that the degree distribution 
            follows a <strong>power-law</strong> (scale-free network).</p>
            <p><strong>What this means:</strong></p>
            <ul>
                <li>Most users have <strong>few connections</strong> (the "long tail")</li>
                <li>A small number of users have <strong>MANY connections</strong> (the "hubs")</li>
                <li>This is typical of social networks: "The rich get richer" phenomenon</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# PAGE 3: NETWORK STATISTICS (Deep Dive)
# ==========================================
elif page == "📈 Network Statistics":
    st.markdown("<div class='big-font'>📈 Advanced Network Statistics</div>", unsafe_allow_html=True)
    st.markdown("<p style='color: #666; font-size: 18px;'>Deep dive into network structure, connectivity, and dynamics</p>", unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["🤝 Social Structure", "🌍 Distance Metrics", "📊 Distribution Analysis", "🔍 Network Properties"])
    
    # --- Tab 1: Social Structure ---
    with tab1:
        st.markdown("### Trust & Reciprocity Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        # Reciprocity
        reciprocity = nx.reciprocity(G)
        col1.metric("🤝 Reciprocity", f"{reciprocity*100:.2f}%", help="Percentage of mutual voting relationships")
        
        # Clustering
        transitivity = nx.transitivity(G)
        col2.metric("🔺 Transitivity", f"{transitivity:.4f}", help="Global clustering coefficient")
        
        # Average clustering coefficient
        avg_clustering = nx.average_clustering(G)
        col3.metric("📊 Avg Clustering", f"{avg_clustering:.4f}", help="Average local clustering coefficient")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='insight-box'>
            <h4>📖 Understanding These Metrics:</h4>
            <ul>
                <li><strong>Reciprocity ({:.2f}%):</strong> Low reciprocity indicates a <strong>hierarchical structure</strong> - 
                users vote for admins, but admins don't necessarily vote back. This is typical of authority networks.</li>
                <li><strong>Transitivity ({:.4f}):</strong> Measures the probability that two of your connections are also connected. 
                A value above 0 indicates clustering, but ours is relatively low due to the hierarchical nature.</li>
                <li><strong>Clustering ({:.4f}):</strong> Shows how tightly nodes cluster together in neighborhoods.</li>
            </ul>
        </div>
        """.format(reciprocity*100, transitivity, avg_clustering), unsafe_allow_html=True)
        
        # Reciprocity visualization
        st.markdown("#### 🔄 Reciprocity Breakdown")
        mutual_edges = sum(1 for u, v in G.edges() if G.has_edge(v, u)) / 2
        one_way_edges = G.number_of_edges() - (mutual_edges * 2)
        
        fig_reciprocity = go.Figure(data=[go.Pie(
            labels=['Mutual Votes', 'One-Way Votes'],
            values=[mutual_edges * 2, one_way_edges],
            hole=.4,
            marker_colors=['#667eea', '#f093fb']
        )])
        fig_reciprocity.update_layout(
            title_text="Distribution of Mutual vs One-Way Votes",
            height=400
        )
        st.plotly_chart(fig_reciprocity, use_container_width=True)

    # --- Tab 2: Distance Metrics ---
    with tab2:
        st.markdown("### 🌍 Small World Analysis")
        st.markdown("Calculating the 'degrees of separation' - how many steps to reach anyone in the network?")
        
        if st.button("🚀 Run Distance Analysis", type="primary"):
            with st.spinner("🔄 Extracting Giant Component & Calculating Paths..."):
                # Extract Giant Component (Undirected view for connectivity)
                U = G.to_undirected()
                largest_cc = max(nx.connected_components(U), key=len)
                subgraph = U.subgraph(largest_cc)
                
                st.success(f"✅ Giant Component extracted: {len(subgraph.nodes())} nodes ({len(subgraph.nodes())/G.number_of_nodes()*100:.1f}% of network)")
                
                # Metrics
                diameter = nx.diameter(subgraph)
                avg_path = nx.average_shortest_path_length(subgraph)
                radius = nx.radius(subgraph)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("🌐 Network Diameter", f"{diameter} steps", help="Longest shortest path in the network")
                col2.metric("📏 Avg Path Length", f"{avg_path:.2f} steps", help="Average distance between any two nodes")
                col3.metric("⭕ Network Radius", f"{radius} steps", help="Minimum eccentricity in the network")
                
                st.markdown("""
                <div class='success-box'>
                    <h4>✅ Small World Confirmed!</h4>
                    <p>With an average path length of <strong>{:.2f} steps</strong>, this network exhibits the 
                    <strong>"small world"</strong> property - any user can reach any other user through just a few intermediaries.</p>
                    <p>This is similar to the famous "6 degrees of separation" in social networks!</p>
                </div>
                """.format(avg_path), unsafe_allow_html=True)

    # --- Tab 3: Distribution ---
    with tab3:
        st.markdown("### 📊 Degree Distribution Analysis")
        
        degrees = [d for n, d in G.degree()]
        
        # Create interactive plotly figure
        fig = go.Figure()
        
        # Histogram
        fig.add_trace(go.Histogram(
            x=degrees,
            nbinsx=50,
            name='Degree Distribution',
            marker_color='#667eea',
            opacity=0.75
        ))
        
        fig.update_layout(
            title="Degree Distribution (Linear Scale)",
            xaxis_title="Degree",
            yaxis_title="Frequency",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Log-Log plot
        from collections import Counter
        degree_counts = Counter(degrees)
        degrees_sorted = sorted(degree_counts.keys())
        counts = [degree_counts[d] for d in degrees_sorted]
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=degrees_sorted,
            y=counts,
            mode='markers',
            marker=dict(size=8, color='#764ba2', opacity=0.6),
            name='Degree Distribution'
        ))
        
        fig2.update_layout(
            title="Degree Distribution (Log-Log Scale - Power Law)",
            xaxis_title="Degree (log scale)",
            yaxis_title="Frequency (log scale)",
            xaxis_type="log",
            yaxis_type="log",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("""
        <div class='warning-box'>
            <strong>⚠️ Power Law Observation:</strong> The approximately linear pattern in the log-log plot confirms 
            a <strong>scale-free network</strong>. This means:<br>
            • Most nodes have few connections (the "masses")<br>
            • A few nodes have many connections (the "hubs")<br>
            • This is characteristic of real-world social networks!
        </div>
        """, unsafe_allow_html=True)
    
    # --- Tab 4: Network Properties ---
    with tab4:
        st.markdown("### 🔍 Additional Network Properties")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Basic Properties")
            st.metric("Nodes", f"{G.number_of_nodes():,}")
            st.metric("Edges", f"{G.number_of_edges():,}")
            st.metric("Density", f"{nx.density(G):.6f}")
            st.metric("Is Directed", "Yes ✓")
            
        with col2:
            st.markdown("#### 🔢 Degree Statistics")
            degrees_list = [d for n, d in G.degree()]
            st.metric("Mean Degree", f"{np.mean(degrees_list):.2f}")
            st.metric("Median Degree", f"{np.median(degrees_list):.0f}")
            st.metric("Std Deviation", f"{np.std(degrees_list):.2f}")
            st.metric("Max Degree", f"{max(degrees_list):,}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Component analysis
        st.markdown("#### 🌐 Component Analysis")
        
        if G.is_directed():
            num_weakly_cc = nx.number_weakly_connected_components(G)
            num_strongly_cc = nx.number_strongly_connected_components(G)
            
            col_a, col_b = st.columns(2)
            col_a.metric("Weakly Connected Components", num_weakly_cc)
            col_b.metric("Strongly Connected Components", num_strongly_cc)
            
            st.info("**Weakly Connected:** Nodes connected by any path (ignoring direction). **Strongly Connected:** Nodes with directed paths in both directions.")

# ==========================================
# PAGE 3: POWER & ROLES (Centrality)
# ==========================================
elif page == "👑 Power & Roles (Centrality)":
    st.markdown("<div class='big-font'>👑 Centrality & User Roles</div>", unsafe_allow_html=True)
    st.markdown("<p style='color: #666; font-size: 18px;'>Identifying the VIPs: Who runs Wikipedia?</p>", unsafe_allow_html=True)
    
    # Caching these heavy calcs
    if "centrality_df" not in st.session_state:
        with st.spinner("🔄 Calculating Centrality Metrics (PageRank, Betweenness, Closeness)..."):
            progress_bar = st.progress(0)
            
            in_degree = nx.in_degree_centrality(G)
            progress_bar.progress(25)
            
            pagerank = nx.pagerank(G, alpha=0.85)
            progress_bar.progress(50)
            
            betweenness = nx.betweenness_centrality(G, k=1000)  # Sample for speed
            progress_bar.progress(75)
            
            # Closeness on a sample for speed
            out_degree = nx.out_degree_centrality(G)
            progress_bar.progress(100)
            
            st.session_state.centrality_df = pd.DataFrame({
                'In-Degree': in_degree,
                'Out-Degree': out_degree,
                'PageRank': pagerank,
                'Betweenness': betweenness
            })
    
    df_metrics = st.session_state.centrality_df
    
    # Explanation
    st.markdown("""
    <div class='insight-box'>
        <h4>🎯 Understanding Centrality Metrics:</h4>
        <ul>
            <li><strong>In-Degree Centrality:</strong> Direct popularity - how many votes you received</li>
            <li><strong>PageRank:</strong> Quality of connections - being voted by important people</li>
            <li><strong>Betweenness Centrality:</strong> Bridge role - how often you connect different groups</li>
            <li><strong>Out-Degree Centrality:</strong> Activity level - how many people you voted for</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Top Rankings with Interactive Charts
    tab1, tab2, tab3, tab4 = st.tabs(["👑 Most Popular", "⭐ Most Influential", "🌉 Best Brokers", "🗳️ Most Active"])
    
    with tab1:
        st.markdown("### 🏆 Top 15 by In-Degree Centrality")
        st.caption("Users with the most direct votes - the most popular/trusted")
        
        top_in = df_metrics.nlargest(15, 'In-Degree')
        top_in_reset = top_in.reset_index()
        top_in_reset.columns = ['User ID', 'In-Degree', 'Out-Degree', 'PageRank', 'Betweenness']
        
        fig1 = px.bar(top_in_reset, x='User ID', y='In-Degree',
                      title='Top 15 Users by In-Degree Centrality',
                      color='In-Degree',
                      color_continuous_scale='Blues',
                      text='In-Degree')
        fig1.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig1.update_layout(height=500)
        st.plotly_chart(fig1, use_container_width=True)
        
        st.dataframe(top_in_reset[['User ID', 'In-Degree']], use_container_width=True, hide_index=True)
    
    with tab2:
        st.markdown("### ⭐ Top 15 by PageRank")
        st.caption("Users with the highest quality connections - true influencers")
        
        top_pr = df_metrics.nlargest(15, 'PageRank')
        top_pr_reset = top_pr.reset_index()
        top_pr_reset.columns = ['User ID', 'In-Degree', 'Out-Degree', 'PageRank', 'Betweenness']
        
        fig2 = px.bar(top_pr_reset, x='User ID', y='PageRank',
                      title='Top 15 Users by PageRank',
                      color='PageRank',
                      color_continuous_scale='Viridis',
                      text='PageRank')
        fig2.update_traces(texttemplate='%{text:.6f}', textposition='outside')
        fig2.update_layout(height=500)
        st.plotly_chart(fig2, use_container_width=True)
        
        st.dataframe(top_pr_reset[['User ID', 'PageRank']], use_container_width=True, hide_index=True)
    
    with tab3:
        st.markdown("### 🌉 Top 15 by Betweenness Centrality")
        st.caption("Users who bridge different communities - the connectors")
        
        top_bt = df_metrics.nlargest(15, 'Betweenness')
        top_bt_reset = top_bt.reset_index()
        top_bt_reset.columns = ['User ID', 'In-Degree', 'Out-Degree', 'PageRank', 'Betweenness']
        
        fig3 = px.bar(top_bt_reset, x='User ID', y='Betweenness',
                      title='Top 15 Users by Betweenness Centrality',
                      color='Betweenness',
                      color_continuous_scale='Plasma',
                      text='Betweenness')
        fig3.update_traces(texttemplate='%{text:.6f}', textposition='outside')
        fig3.update_layout(height=500)
        st.plotly_chart(fig3, use_container_width=True)
        
        st.dataframe(top_bt_reset[['User ID', 'Betweenness']], use_container_width=True, hide_index=True)
    
    with tab4:
        st.markdown("### 🗳️ Top 15 by Out-Degree Centrality")
        st.caption("Most active voters - highly engaged users")
        
        top_out = df_metrics.nlargest(15, 'Out-Degree')
        top_out_reset = top_out.reset_index()
        top_out_reset.columns = ['User ID', 'In-Degree', 'Out-Degree', 'PageRank', 'Betweenness']
        
        fig4 = px.bar(top_out_reset, x='User ID', y='Out-Degree',
                      title='Top 15 Users by Out-Degree Centrality',
                      color='Out-Degree',
                      color_continuous_scale='Sunset',
                      text='Out-Degree')
        fig4.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig4.update_layout(height=500)
        st.plotly_chart(fig4, use_container_width=True)
        
        st.dataframe(top_out_reset[['User ID', 'Out-Degree']], use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    
    # THE ROLES TABLE (Insight)
    st.markdown("### 🧠 Strategic User Classification")
    st.markdown("Based on centrality metrics, we can categorize users into strategic roles:")
    
    roles_data = {
        "Role": ["👑 Authorities", "🌉 Brokers", "⭐ All-Rounders", "🗳️ Active Voters"],
        "Characteristics": [
            "High In-Degree & PageRank, Lower Betweenness",
            "High Betweenness, Moderate other metrics",
            "High across ALL metrics - rare super-users",
            "High Out-Degree, Lower In-Degree"
        ],
        "Network Function": [
            "Provide legitimacy & make key decisions",
            "Connect communities & prevent fragmentation",
            "Maintain system stability - critical nodes",
            "Drive engagement & participation"
        ],
        "Impact": [
            "High influence on community decisions",
            "Critical for network cohesion",
            "Maximum structural importance",
            "Essential for network activity"
        ]
    }
    
    df_roles = pd.DataFrame(roles_data)
    st.dataframe(df_roles, use_container_width=True, hide_index=True)
    
    st.markdown("""
    <div class='success-box'>
        <strong>💡 Key Insight:</strong> The network has a clear hierarchy with different user roles. 
        <strong>Authorities</strong> hold the most power, while <strong>Brokers</strong> keep the network connected. 
        This division of roles is typical in organizational and social networks.
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# PAGE 4: VISUALIZATIONS
# ==========================================
elif page == "🎨 Visualizations (Graphs)":
    st.markdown("<div class='big-font'>🎨 Network Visualizations</div>", unsafe_allow_html=True)
    st.markdown("<p style='color: #666; font-size: 18px;'>Visual exploration of network structure and patterns</p>", unsafe_allow_html=True)
    
    tab_v1, tab_v2, tab_v3 = st.tabs(["🕸️ Network Graph", "🔥 Matrix Heatmap", "📍 Interactive 3D"])
    
    # --- Graph Viz ---
    with tab_v1:
        st.markdown("### 🕸️ Top 100 Elite Users Network")
        st.caption("Nodes sized by votes received, colored by community, labeled with user IDs")
        
        # Size selector
        top_n = st.slider("Select number of top nodes to visualize:", 20, 150, 100, 10)
        
        # Filter Top N
        top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:top_n]
        nodes_list = [n for n, d in top_nodes]
        subgraph = G.subgraph(nodes_list)
        
        # Layout
        layout_type = st.selectbox("Select Layout Algorithm:", 
                                    ["Spring (Force-directed)", "Circular", "Kamada-Kawai"])
        
        if layout_type == "Spring (Force-directed)":
            pos = nx.spring_layout(subgraph, seed=42, k=0.5, iterations=50)
        elif layout_type == "Circular":
            pos = nx.circular_layout(subgraph)
        else:
            pos = nx.kamada_kawai_layout(subgraph)
        
        # Draw
        fig, ax = plt.subplots(figsize=(16, 16))
        
        # Community colors
        try:
            communities = list(greedy_modularity_communities(subgraph.to_undirected()))
            color_map = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    color_map[node] = i
            node_colors = [color_map.get(n, 0) for n in subgraph.nodes()]
        except:
            node_colors = ['#667eea'] * len(subgraph.nodes())
        
        # Draw edges first (in background)
        nx.draw_networkx_edges(subgraph, pos, alpha=0.15, edge_color='gray', 
                               width=0.5, arrows=True, arrowsize=8, 
                               arrowstyle='->', connectionstyle='arc3,rad=0.1')
        
        # Draw nodes
        node_sizes = [subgraph.in_degree(n) * 50 + 100 for n in subgraph.nodes()]
        nx.draw_networkx_nodes(subgraph, pos, node_size=node_sizes,
                               node_color=node_colors, cmap=plt.cm.tab10, 
                               alpha=0.8, edgecolors='black', linewidths=1.5)
        
        # Draw labels for top 30 nodes only
        if top_n <= 50:
            nx.draw_networkx_labels(subgraph, pos, font_size=9, 
                                   font_color='black', font_weight='bold')
        else:
            top_30_nodes = nodes_list[:30]
            labels = {n: n for n in top_30_nodes if n in subgraph.nodes()}
            nx.draw_networkx_labels(subgraph, pos, labels=labels, font_size=8,
                                   font_color='black', font_weight='bold')
        
        plt.title(f"Network Visualization: Top {top_n} Users", fontsize=20, fontweight='bold', pad=20)
        plt.axis('off')
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
        <div class='success-box'>
            <strong>🔍 Observations:</strong>
            <ul>
                <li><strong>Dense Center:</strong> The "Rich Club" effect - highly connected elite nodes cluster together</li>
                <li><strong>Color Clusters:</strong> Different colors represent distinct communities even among top users</li>
                <li><strong>Node Size:</strong> Larger nodes received more votes (higher in-degree)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # --- Heatmap Viz ---
    with tab_v2:
        st.markdown("### 🔥 Adjacency Matrix Heatmap")
        st.caption("Visual representation of voting patterns - who voted for whom")
        
        matrix_size = st.slider("Matrix size (top N users):", 20, 50, 30, 5)
        
        top_n_matrix = sorted(G.degree, key=lambda x: x[1], reverse=True)[:matrix_size]
        nodes_matrix = [n for n, d in top_n_matrix]
        sub_matrix = G.subgraph(nodes_matrix)
        matrix = nx.to_pandas_adjacency(sub_matrix, dtype=int)
        
        fig2, ax2 = plt.subplots(figsize=(14, 12))
        sns.heatmap(matrix, cmap="RdYlBu_r", cbar_kws={'label': 'Vote (1=Yes, 0=No)'}, 
                   square=True, linewidths=0.3, linecolor='white',
                   annot=False, fmt='d', cbar=True)
        plt.xlabel("Candidate (Voted For)", fontsize=12, fontweight='bold')
        plt.ylabel("Voter (Voting User)", fontsize=12, fontweight='bold')
        plt.title(f"Voting Matrix: Top {matrix_size} Users", fontsize=16, fontweight='bold', pad=15)
        plt.tight_layout()
        st.pyplot(fig2)
        
        st.markdown("""
        <div class='insight-box'>
            <strong>📊 How to Read:</strong>
            <ul>
                <li><strong>Rows:</strong> Voters (who is voting)</li>
                <li><strong>Columns:</strong> Candidates (who receives votes)</li>
                <li><strong>Red Cells:</strong> A vote exists from row user to column user</li>
                <li><strong>Blue Cells:</strong> No vote relationship</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # --- 3D Interactive ---
    with tab_v3:
        st.markdown("### 📍 Interactive 3D Network Visualization")
        st.caption("Explore the network in 3D space - rotate, zoom, and interact!")
        
        n_nodes_3d = st.slider("Number of nodes for 3D visualization:", 30, 100, 50, 10)
        
        # Get top nodes
        top_3d = sorted(G.degree, key=lambda x: x[1], reverse=True)[:n_nodes_3d]
        nodes_3d = [n for n, d in top_3d]
        sub_3d = G.subgraph(nodes_3d)
        
        # 3D spring layout
        pos_3d = nx.spring_layout(sub_3d, dim=3, seed=42, k=0.5)
        
        # Extract coordinates
        x_nodes = [pos_3d[node][0] for node in sub_3d.nodes()]
        y_nodes = [pos_3d[node][1] for node in sub_3d.nodes()]
        z_nodes = [pos_3d[node][2] for node in sub_3d.nodes()]
        
        # Node sizes based on degree
        node_sizes_3d = [sub_3d.degree(n) * 3 for n in sub_3d.nodes()]
        
        # Create edges
        edge_x = []
        edge_y = []
        edge_z = []
        
        for edge in sub_3d.edges():
            x0, y0, z0 = pos_3d[edge[0]]
            x1, y1, z1 = pos_3d[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])
        
        # Create 3D plot
        fig_3d = go.Figure()
        
        # Add edges
        fig_3d.add_trace(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(color='rgba(125,125,125,0.3)', width=1),
            hoverinfo='none',
            showlegend=False
        ))
        
        # Add nodes
        fig_3d.add_trace(go.Scatter3d(
            x=x_nodes, y=y_nodes, z=z_nodes,
            mode='markers+text',
            marker=dict(
                size=node_sizes_3d,
                color=node_sizes_3d,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Degree"),
                line=dict(color='white', width=0.5)
            ),
            text=[str(node) for node in sub_3d.nodes()],
            textposition="top center",
            textfont=dict(size=8),
            hovertemplate='<b>User %{text}</b><br>Degree: %{marker.size:.0f}<extra></extra>',
            showlegend=False
        ))
        
        fig_3d.update_layout(
            title=f"3D Network Visualization: Top {n_nodes_3d} Users",
            scene=dict(
                xaxis=dict(showbackground=False, showticklabels=False, title=''),
                yaxis=dict(showbackground=False, showticklabels=False, title=''),
                zaxis=dict(showbackground=False, showticklabels=False, title='')
            ),
            height=700,
            hovermode='closest'
        )
        
        st.plotly_chart(fig_3d, use_container_width=True)
        
        st.info("💡 **Tip:** Click and drag to rotate, scroll to zoom, double-click to reset view!")

# ==========================================
# PAGE 5: COMMUNITY DETECTION
# ==========================================
elif page == "🌐 Community Detection":
    st.markdown("<div class='big-font'>🌐 Community Detection</div>", unsafe_allow_html=True)
    st.markdown("<p style='color: #666; font-size: 18px;'>Discovering natural groupings and sub-communities in the network</p>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='insight-box'>
        <h4>🎯 What are Communities?</h4>
        <p>Communities are groups of nodes that are more densely connected to each other than to the rest of the network. 
        In social networks, these often represent natural clusters of like-minded individuals or organizational units.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Run community detection
    if st.button("🚀 Detect Communities", type="primary"):
        with st.spinner("🔄 Running community detection algorithms..."):
            # Convert to undirected for community detection
            G_undirected = G.to_undirected()
            
            # Greedy modularity communities
            communities = list(greedy_modularity_communities(G_undirected))
            
            # Calculate modularity
            modularity = nx.community.modularity(G_undirected, communities)
            
            st.success(f"✅ Found {len(communities)} communities with modularity score of {modularity:.4f}")
            
            # Display community stats
            col1, col2, col3 = st.columns(3)
            col1.metric("🏘️ Total Communities", len(communities))
            col2.metric("📊 Modularity Score", f"{modularity:.4f}", help="Higher is better (0-1 scale)")
            col3.metric("👥 Largest Community", max(len(c) for c in communities))
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Community sizes
            st.markdown("### 📊 Community Size Distribution")
            
            community_sizes = [len(c) for c in communities]
            community_df = pd.DataFrame({
                'Community ID': range(1, len(communities) + 1),
                'Size': community_sizes
            }).sort_values('Size', ascending=False).reset_index(drop=True)
            
            fig_comm = px.bar(community_df.head(20), x='Community ID', y='Size',
                             title='Top 20 Communities by Size',
                             color='Size',
                             color_continuous_scale='Turbo',
                             text='Size')
            fig_comm.update_traces(textposition='outside')
            fig_comm.update_layout(height=500)
            st.plotly_chart(fig_comm, use_container_width=True)
            
            # Show top communities
            st.markdown("### 🏆 Largest Communities")
            st.dataframe(community_df.head(10), use_container_width=True, hide_index=True)
            
            # Visualize communities
            st.markdown("### 🎨 Community Visualization")
            st.caption("Top 100 nodes colored by community membership")
            
            # Get top 100 nodes
            top_100 = sorted(G.degree, key=lambda x: x[1], reverse=True)[:100]
            nodes_100 = [n for n, d in top_100]
            sub_100 = G.subgraph(nodes_100).to_undirected()
            
            # Get communities for these nodes
            node_to_community = {}
            for idx, comm in enumerate(communities):
                for node in comm:
                    if node in nodes_100:
                        node_to_community[node] = idx
            
            # Layout and draw
            pos = nx.spring_layout(sub_100, seed=42, k=0.5, iterations=50)
            
            fig_viz, ax = plt.subplots(figsize=(16, 14))
            
            # Get colors
            node_colors = [node_to_community.get(n, -1) for n in sub_100.nodes()]
            
            nx.draw_networkx_edges(sub_100, pos, alpha=0.2, edge_color='gray', width=0.5)
            nx.draw_networkx_nodes(sub_100, pos, 
                                  node_size=[sub_100.degree(n) * 40 + 100 for n in sub_100.nodes()],
                                  node_color=node_colors, cmap=plt.cm.tab20,
                                  alpha=0.8, edgecolors='black', linewidths=1.5)
            
            plt.title("Community Structure: Top 100 Users", fontsize=20, fontweight='bold', pad=20)
            plt.axis('off')
            plt.tight_layout()
            st.pyplot(fig_viz)
            
            st.markdown(f"""
            <div class='success-box'>
                <strong>🎯 Key Findings:</strong>
                <ul>
                    <li>The network naturally divides into <strong>{len(communities)} communities</strong></li>
                    <li>Modularity score of <strong>{modularity:.4f}</strong> indicates {'strong' if modularity > 0.4 else 'moderate'} community structure</li>
                    <li>Largest community contains <strong>{max(community_sizes)} members</strong></li>
                    <li>Different colors in the visualization represent different communities</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
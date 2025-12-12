# Wiki-Vote Analytics Dashboard

A comprehensive social network analysis dashboard for Wikipedia voting data, built with Streamlit.

## 🌟 Features

- **📊 Node Degree Analysis** - Analyze user connections and voting patterns
- **📈 Network Statistics** - Deep dive into network structure and metrics
- **👑 Centrality Analysis** - Identify influential users and power structures
- **🎨 Interactive Visualizations** - 2D and 3D network graphs
- **🌐 Community Detection** - Discover natural groupings in the network

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone or download this repository

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Make sure `Wiki-Vote.txt` is in the project directory

4. Run the app:

```bash
streamlit run main.py
```

The app will open in your browser at `http://localhost:8501`

## 📦 Dataset

This project uses the Wikipedia Voting Network dataset:

- **Nodes**: 7,115 Wikipedia users
- **Edges**: 103,689 voting relationships
- **Type**: Directed graph

Place the `Wiki-Vote.txt` file in the project root directory.

## 🌐 Deployment Options

### Option 1: Streamlit Cloud (Recommended - FREE)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click "New app"
5. Select your repository and branch
6. Set main file path: `main.py`
7. Click "Deploy"

**Note**: Make sure `Wiki-Vote.txt` is included in your repository.

### Option 2: Heroku

1. Create `Procfile`:

```
web: streamlit run main.py --server.port=$PORT --server.address=0.0.0.0
```

2. Deploy:

```bash
heroku create your-app-name
git push heroku main
```

### Option 3: Docker

1. Create `Dockerfile`:

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "main.py"]
```

2. Build and run:

```bash
docker build -t wiki-vote-app .
docker run -p 8501:8501 wiki-vote-app
```

## 📊 Project Structure

```
.
├── main.py                 # Main Streamlit application
├── main.ipynb             # Jupyter notebook with analysis
├── Wiki-Vote.txt          # Dataset (required)
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── .streamlit/
    └── config.toml       # Streamlit configuration
```

## 🛠️ Technology Stack

- **Streamlit** - Web framework
- **NetworkX** - Graph analysis
- **Plotly** - Interactive visualizations
- **Matplotlib/Seaborn** - Static visualizations
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing

## 📝 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

- Dataset: Stanford Network Analysis Project (SNAP)
- Wikipedia Voting Network

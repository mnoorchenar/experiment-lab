---
title: experiment-lab
colorFrom: blue
colorTo: indigo
sdk: docker
---

<div align="center">

<h1>🧪 ExperimentLab</h1>
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&duration=3000&pause=1000&color=4f46e5&center=true&vCenter=true&width=700&lines=Power+Analysis+%26+Sample+Size+Calculator;A%2FB+Testing+with+Practical+Significance;Two-Factor+DoE+%26+Interaction+Analysis;Sequential+Testing+%26+Peeking+Problem+Demo" alt="Typing SVG"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3b82f6?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-4f46e5?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-3b82f6?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Status](https://img.shields.io/badge/Status-Active-22c55e?style=for-the-badge)](#)

<br/>

[![Live Demo](https://img.shields.io/badge/%E2%96%B6_Live_Demo-Hugging_Face_Space-ffcc00?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/mnoorchenar/experiment-lab)
[![Source Code](https://img.shields.io/badge/Source_Code-GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/mnoorchenar/experiment-lab)

<br/>

**🧪 ExperimentLab** — An interactive statistical experimentation suite covering sample size planning, A/B test analysis, two-factor factorial design, and sequential testing — built to demonstrate the full experimentation workflow from pre-experiment planning to post-experiment decision-making.

<br/>

---

</div>

## Table of Contents

- [Features](#-features)
- [Architecture](#️-architecture)
- [Getting Started](#-getting-started)
- [Docker Deployment](#-docker-deployment)
- [Dashboard Modules](#-dashboard-modules)
- [ML Models](#-ml-models)
- [Project Structure](#-project-structure)
- [Author](#-author)
- [Contributing](#-contributing)
- [Disclaimer](#️-disclaimer)
- [License](#-license)

---

## ✨ Features

<table>
  <tr>
    <td>📐 <b>Power Calculator</b></td>
    <td>Compute required sample sizes and visualize power curves with configurable MDE, α, and power targets</td>
  </tr>
  <tr>
    <td>⚖️ <b>A/B Test Analyzer</b></td>
    <td>Two-proportion z-test and Welch's t-test with both statistical and practical significance verdicts</td>
  </tr>
  <tr>
    <td>🔬 <b>Two-Factor DoE</b></td>
    <td>Full factorial ANOVA with interaction plots — detects whether two factors interact with each other</td>
  </tr>
  <tr>
    <td>📈 <b>Sequential Testing Demo</b></td>
    <td>Simulates the peeking problem and shows how O'Brien-Fleming boundaries fix false positive inflation</td>
  </tr>
  <tr>
    <td>🐳 <b>Containerized Deployment</b></td>
    <td>Docker-first architecture, cloud-ready and scalable</td>
  </tr>
</table>

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    ExperimentLab                         │
│                                                         │
│  ┌───────────┐    ┌───────────┐    ┌───────────────┐  │
│  │  User     │───▶│  SciPy /  │───▶│   Flask API   │  │
│  │  Inputs   │    │  NumPy    │    │   Backend     │  │
│  └───────────┘    └───────────┘    └───────┬───────┘  │
│                                            │           │
│                                   ┌────────▼────────┐  │
│                                   │  Plotly Charts  │  │
│                                   │   (Browser)     │  │
│                                   └─────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- Git

### Local Installation

```bash
# 1. Clone the repository
git clone https://github.com/mnoorchenar/experiment-lab.git
cd experiment-lab

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env with your settings

# 5. Run the application
python app.py
```

Open your browser at `http://localhost:7860` 🎉

---

## 🐳 Docker Deployment

```bash
# Build and run with Docker Compose
docker compose up --build

docker build -t experiment-lab .

docker run -p 7860:7860 experiment-lab
```

---

## 📊 Dashboard Modules

| Module | Description | Status |
|--------|-------------|--------|
| 📐 Power Calculator | Sample size planning with power curves and MDE trade-off visualizations | ✅ Live |
| ⚖️ A/B Test Analyzer | Statistical & practical significance, CI visualization, distribution overlap | ✅ Live |
| 🔬 Two-Factor DoE | Factorial ANOVA table, interaction plots, main effects analysis | ✅ Live |
| 📈 Sequential Testing | Peeking problem simulation with OBF boundary comparison | ✅ Live |
| 📊 Interactive Visualizations | All charts are interactive Plotly — hoverable, zoomable, downloadable | ✅ Live |
| 📱 Mobile Responsive | Sidebar navigation with responsive layout for all screen sizes | ✅ Live |

---

## 🧠 ML Models

```python
# Core statistical methods used in ExperimentLab
models = {
    "power_analysis":    "Two-proportion z-test power formula (Fleiss, 1981)",
    "ab_proportions":    "Two-proportion z-test with pooled standard error",
    "ab_means":          "Welch's t-test (unequal variances) + Cohen's d",
    "two_factor_anova":  "Two-way ANOVA with interaction (balanced & unbalanced)",
    "sequential_testing":"O'Brien-Fleming alpha-spending boundary (z_α / √t)"
}
```

---

## 📁 Project Structure

```
experiment-lab/
│
├── 📂 routes/                    # Flask blueprints — one per module
│   ├── __init__.py
│   ├── ab_test.py
│   ├── doe.py
│   ├── power.py
│   └── sequential.py
├── 📂 static/                    # CSS, JavaScript & assets
│   ├── css/
│   └── js/
├── 📂 templates/                 # Jinja2 HTML templates
│   ├── ab_test.html
│   ├── base.html
│   ├── doe.html
│   ├── power.html
│   └── sequential.html
├── 📄 .env.example               # Environment variable template
├── 📄 Dockerfile                 # Container definition (port 7860)
├── 📄 app.py                     # Application entry point
├── 📄 docker-compose.yml         # Multi-service orchestration
└── 📄 requirements.txt           # Python dependencies
```

---

## 👨‍💻 Author

<div align="center">

<table>
<tr>
<td align="center" width="100%">

<img src="https://avatars.githubusercontent.com/mnoorchenar" width="120" style="border-radius:50%; border: 3px solid #4f46e5;" alt="Mohammad Noorchenarboo"/>

<h3>Mohammad Noorchenarboo</h3>

<code>Data Scientist</code> &nbsp;|&nbsp; <code>AI Researcher</code> &nbsp;|&nbsp; <code>Biostatistician</code>

📍 &nbsp;Ontario, Canada &nbsp;&nbsp; 📧 &nbsp;[mohammadnoorchenarboo@gmail.com](mailto:mohammadnoorchenarboo@gmail.com)

──────────────────────────────────────

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mnoorchenar)&nbsp;
[![Personal Site](https://img.shields.io/badge/Website-mnoorchenar.github.io-4f46e5?style=for-the-badge&logo=githubpages&logoColor=white)](https://mnoorchenar.github.io/)&nbsp;
[![HuggingFace](https://img.shields.io/badge/HuggingFace-ffcc00?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/mnoorchenar)&nbsp;
[![Google Scholar](https://img.shields.io/badge/Scholar-4285F4?style=for-the-badge&logo=googlescholar&logoColor=white)](https://scholar.google.ca/citations?user=nn_Toq0AAAAJ&hl=en)&nbsp;
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/mnoorchenar)

</td>
</tr>
</table>

</div>

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

---

## ⚠️ Disclaimer

> [!WARNING]
> This project is developed strictly for educational and research purposes and does not constitute professional advice of any kind. All datasets used are either synthetically generated or publicly available — no real user data is stored. This software is provided "as is" without warranty of any kind; use at your own risk.

---

## 📜 License

Distributed under the **MIT License**.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:3b82f6,100:4f46e5&height=120&section=footer&text=Made%20with%20%E2%9D%A4%EF%B8%8F%20by%20Mohammad%20Noorchenarboo&fontColor=ffffff&fontSize=18&fontAlignY=80" width="100%"/>

[![GitHub Stars](https://img.shields.io/github/stars/mnoorchenar/experiment-lab?style=social)](https://github.com/mnoorchenar/experiment-lab/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/mnoorchenar/experiment-lab?style=social)](https://github.com/mnoorchenar/experiment-lab/fork)

<sub>The name "ExperimentLab" is used purely for academic and research purposes. Any similarity to existing product names or trademarks is entirely coincidental. This project has no affiliation with any commercial entity.</sub>

</div>

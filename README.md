# Insurance Actuarial Pricing & Loss Reserving Optimizer

[![Actuarial](https://img.shields.io/badge/Actuarial-Chain_Ladder_IBNR-4B0082?style=for-the-badge)](https://en.wikipedia.org/wiki/Chain-ladder_method) [![GLM](https://img.shields.io/badge/Pricing-Tweedie_GLM-008080?style=for-the-badge)](https://scikit-learn.org/) [![Python](https://img.shields.io/badge/Python-Actuarial_Science-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Author](https://img.shields.io/badge/Author-Abdussatar-E50914?style=for-the-badge&logo=github&logoColor=white)](https://github.com/abdussatarkhan)

> **An actuarial intelligence suite implementing Chain-Ladder / Bornhuetter-Ferguson claims loss reserving triangles and Tweedie Generalized Linear Models (GLM) for pure premium insurance pricing.**

---

## 🏛️ System Architecture

```mermaid
graph TD
    Claims[Historical Policy Claims & Exposure Data] --> Triangle[Chain-Ladder Claims Development Triangle]
    Triangle --> IBNR[IBNR Reserve Estimation]
    Claims --> Tweedie[Tweedie GLM Frequency-Severity Modeling]
    Tweedie --> RateCard[Risk-Differentiated Tariff Rate Card]
```

---

## 🌟 Key Features & Capabilities

- **Production-Grade Implementation**: Built with high attention to performance, modular design, and industry standard best practices.
- **Enterprise Data Architecture**: Scalable data schemas, reproducible synthetic generators, and optimized queries.
- **Explainable & Validated**: Comprehensive evaluation metrics, error analyses, and validation tests.
- **Comprehensive Tech Stack**: `Python` `Statsmodels` `Scikit-Learn` `Actuarial Science` `Pandas`.

---

## 📊 Visual Preview & Analysis

<div align="center">

![insurance-underwriting-optimizer preview](images/chain_ladder_triangle.png)

</div>

---

## 🚀 Quickstart & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/abdussatarkhan/insurance-underwriting-optimizer.git
cd insurance-underwriting-optimizer
```

### 2. Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies (if requirements.txt exists)
pip install -r requirements.txt
```

---

## 👨‍💻 Author & Profile

Built and maintained by **Abdussatar** ([@abdussatarkhan](https://github.com/abdussatarkhan)).  
For technical discussions, collaboration, or queries, feel free to reach out via [LinkedIn](https://www.linkedin.com/in/abdus-satar-5150813b5/) or [GitHub](https://github.com/abdussatarkhan).

---

## 📜 License

This project is licensed under the **MIT License** — see the LICENSE file for details.

# Insurance Actuarial Pricing & Loss Reserving Optimizer

[![CI](https://github.com/abdussatarkhan/insurance-underwriting-optimizer/actions/workflows/ci.yml/badge.svg)](https://github.com/abdussatarkhan/insurance-underwriting-optimizer/actions)
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

<div align="center">

[![Daily Streak](https://img.shields.io/badge/Daily%20Streak-Active%20%F0%9F%94%A5-brightgreen?style=flat-square&logo=github)](https://github.com/abdussatarkhan)
[![Master Portfolio](https://img.shields.io/badge/Portfolio-50%2B%20Enterprise%20Projects-0e75b6?style=flat-square&logo=github)](https://github.com/abdussatarkhan/abdussatarkhan)
[![Author: Abdussatar](https://img.shields.io/badge/Author-Abdussatar-24292e?style=flat-square&logo=github)](https://github.com/abdussatarkhan)

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

## 🗺️ Roadmap & Upcoming Features

- [x] Chain-Ladder & Bornhuetter-Ferguson loss reserving triangles
- [x] Tweedie GLM pure premium frequency-severity pricing
- [ ] Machine Learning gradient boosted Tweedie models (LightGBM)
- [ ] Dynamic retention survival hazard modeling
- [ ] Interactive actuarial rate-filing Excel/PDF exporter

---

## 👨‍💻 Author & Profile

Built and maintained by **Abdussatar** ([@abdussatarkhan](https://github.com/abdussatarkhan)).  
For technical discussions, collaboration, or queries, feel free to reach out via [LinkedIn](https://www.linkedin.com/in/abdus-satar-5150813b5/) or [GitHub](https://github.com/abdussatarkhan).

---

## 📜 License

This project is licensed under the **MIT License** — see the LICENSE file for details.


---

<div align="center">

### 👨‍💻 Maintained by [Abdussatar (@abdussatarkhan)](https://github.com/abdussatarkhan)
Part of the **[Master Enterprise Data Analytics & AI Portfolio](https://github.com/abdussatarkhan/abdussatarkhan)**.

⭐ If you find this repository valuable, consider dropping a star! ⭐

</div>

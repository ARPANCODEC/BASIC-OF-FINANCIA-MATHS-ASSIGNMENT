📊 Financial Mathematics Assignment — Interactive Streamlit App

This repository contains an interactive Streamlit web application that solves a complete Financial Mathematics assignment using clear formulas, fully-worked calculations, binomial pricing trees, and visualizations.
The app displays each question, explains the mathematical methodology, and generates step-by-step answers dynamically based on user inputs.

🚀 Features
1️⃣ Mortgage Amortization (Question 1)

Computes monthly EMI using the Present Value of an Annuity formula.

Generates balance after any payment month using retrospective formula.

Includes an interactive amortization table (first 24 months).

Provides exact formulas, annotations, and explanation of every calculation.

2️⃣ American Put Option Pricing (Question 2)

Prices a 5-month American put using a 5-step CRR binomial tree.

Displays:

Up/down factors

Risk-neutral probability

Discount factor

Full backward induction

Identifies nodes where early exercise is optimal.

Implements the Control Variate Technique using:

Black–Scholes European Put (closed form)

Binomial European Put

Computes an improved American put price with reduced discretization error.

Optional tree heatmap visualization.

3️⃣ Efficient Portfolios & Tangency Portfolio (Question 3)

Computes the Tangency Portfolio using:

Expected returns

3×3 covariance matrix of risky assets

Inverse-covariance weighting

Calculates:

Sharpe Ratio

Expected return & standard deviation of the tangency portfolio

Generates the Capital Market Line graphically.

Allows the user to:

Choose leverage up to 200%

Enter target return and compute required weights

View optimal allocation among risky assets and the risk-free asset.

4️⃣ Capital Market Line (CML) Analysis (Question 4)

Derives the CML equation.

Computes portfolio return for any RF/Market combination.

Determines portfolio weights required to reach a target return (including leverage/shorting warnings).

🧮 Mathematics Covered

✔ Loan amortization theory
✔ Time value of money
✔ Option pricing (CRR model)
✔ Black–Scholes model
✔ Risk-neutral valuation
✔ Early exercise logic (American options)
✔ Markowitz portfolio theory
✔ Tangency portfolio
✔ Capital Market Line (CML)
✔ Sharpe ratio optimization

🛠 Technologies Used

Python

Streamlit (UI framework)

NumPy / Pandas / Math

SciPy.stats (Normal CDF)

Matplotlib (graphs & tree visualizations)

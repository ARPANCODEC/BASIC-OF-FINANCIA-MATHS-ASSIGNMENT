import streamlit as st
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import plotly.graph_objects as go
import plotly.express as px

# --- Page config and styling ---
st.set_page_config(
    page_title="Financial Mathematics Assignment Solutions", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2e86ab;
        border-bottom: 2px solid #2e86ab;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .subsection-header {
        font-size: 1.4rem;
        color: #3a8fc5;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .formula-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2e86ab;
        margin: 1rem 0;
        font-family: "Courier New", monospace;
    }
    .result-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #2e86ab;
        margin: 1rem 0;
    }
    .annotation {
        font-style: italic;
        color: #666;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    .tree-node {
        background-color: #f0f8ff;
        border: 1px solid #87ceeb;
        border-radius: 5px;
        padding: 5px;
        margin: 2px;
        font-size: 0.8rem;
    }
    .calculation-step {
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        border-left: 3px solid #2e86ab;
    }
    .early-exercise {
        background-color: #ffebee;
        border: 2px solid #f44336;
    }
    .hold-node {
        background-color: #e8f5e8;
        border: 1px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">Financial Mathematics Assignment Solutions</div>', unsafe_allow_html=True)

st.markdown("""
This interactive application provides detailed solutions to financial mathematics problems with step-by-step calculations, 
proper mathematical formulations, and visualizations.
""")

st.write("---")

# ---------------- QUESTION 1 ----------------
st.markdown('<div class="section-header">Question 1 — Mortgage Amortization</div>', unsafe_allow_html=True)

st.markdown("""
**Problem Statement:**  
Suppose you take a mortgage loan for an amount \( L \) that is to be paid back over \( n \) months with equal payments of \( A \) at the end of each month. 
The interest rate for the loan is \( r \) per month, compounded monthly.

**(i)** Find \( A \) in terms of \( L, n, r \).  
**(ii)** Find the remaining balance after payment at the end of month \( j \).  

**Check your result with:** \( L = \\$100,000 \), \( n = 360 \) months, annual interest rate = 9% compounded monthly.
""")

with st.expander("🔍 Show Detailed Solution & Calculator", expanded=False):
    st.markdown('<div class="subsection-header">Mathematical Formulation</div>', unsafe_allow_html=True)
    
    st.markdown("""
    **Key Formulas:**
    
    1. **Present Value of Annuity (to find monthly payment):**
    """)
    st.latex(r"L = A \times \frac{1 - (1 + r)^{-n}}{r}")
    st.markdown('<div class="annotation">This formula equates the loan amount to the present value of all future payments.</div>', unsafe_allow_html=True)
    
    st.markdown("Solving for A:")
    st.latex(r"A = L \times \frac{r}{1 - (1 + r)^{-n}}")
    
    st.markdown("""
    2. **Remaining Balance (Retrospective Method):**
    """)
    st.latex(r"B_j = L \times (1 + r)^j - A \times \frac{(1 + r)^j - 1}{r}")
    st.markdown('<div class="annotation">This calculates the remaining balance as the future value of original loan minus the future value of payments made.</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="subsection-header">Interactive Calculator</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("**Loan Parameters**")
        L = st.number_input("Loan Amount \( L \) ($)", value=100000.0, min_value=0.0, step=1000.0, format="%.2f", key="q1_L")
        n = st.number_input("Number of Months \( n \)", value=360, min_value=1, step=1, key="q1_n")
        annual_rate = st.number_input("Annual Interest Rate (%)", value=9.0, min_value=0.0, step=0.01, format="%.4f", key="q1_rate")
    
    with col2:
        st.markdown("**Calculation Parameters**")
        r = annual_rate / 100.0 / 12.0
        st.info(f"Monthly interest rate: {r:.6f}")
        j = st.number_input("Check balance after month \( j \)", value=120, min_value=0, max_value=n, step=1, key="q1_j")
    
    with col3:
        st.markdown("**Results**")
        # Compute monthly payment A
        if r == 0:
            A = L / n
        else:
            A = (L * r) / (1 - (1 + r) ** (-n))
        
        # Compute remaining balance after j payments
        if r == 0:
            B_j = L - A * j
        else:
            B_j = L * (1 + r) ** j - A * (((1 + r) ** j - 1) / r)
        
        st.metric("Monthly Payment \( A \)", f"${A:,.2f}")
        st.metric(f"Balance after {j} months", f"${B_j:,.2f}")
    
    # Detailed calculations
    st.markdown('<div class="subsection-header">Step-by-Step Calculation</div>', unsafe_allow_html=True)
    
    if r > 0:
        st.markdown('<div class="calculation-step">', unsafe_allow_html=True)
        st.write(f"**Monthly Payment Calculation:**")
        st.write(f"1. Monthly interest rate: \( r = {annual_rate}\% / 12 / 100 = {r:.6f} \)")
        st.write(f"2. \( (1 + r) = {1 + r:.6f} \)")
        st.write(f"3. \( (1 + r)^{-n} = {((1 + r)**(-n)):.8f} \)")
        st.write(f"4. \( 1 - (1 + r)^{-n} = {1 - ((1 + r)**(-n)):.8f} \)")
        st.write(f"5. \( A = L \\times \\frac{{r}}{{1 - (1 + r)^{-n}}} = {L:,.2f} \\times \\frac{{{r:.6f}}}{{{1 - ((1 + r)**(-n)):.8f}}} = {A:,.2f} \)")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="calculation-step">', unsafe_allow_html=True)
        st.write(f"**Remaining Balance after {j} payments:**")
        st.write(f"1. \( (1 + r)^j = {((1 + r)**j):.6f} \)")
        st.write(f"2. Future value of loan: \( L \\times (1 + r)^j = {L:,.2f} \\times {((1 + r)**j):.6f} = {L * ((1 + r)**j):,.2f} \)")
        st.write(f"3. Future value of payments: \( A \\times \\frac{{(1 + r)^j - 1}}{{r}} = {A:,.2f} \\times \\frac{{{((1 + r)**j):.6f} - 1}}{{{r:.6f}}} = {A * (((1 + r)**j - 1) / r):,.2f} \)")
        st.write(f"4. \( B_{j} = {L * ((1 + r)**j):,.2f} - {A * (((1 + r)**j - 1) / r):,.2f} = {B_j:,.2f} \)")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Amortization schedule
    st.markdown('<div class="subsection-header">Amortization Schedule</div>', unsafe_allow_html=True)
    if st.checkbox("Show amortization schedule (first 24 months)", key="q1_amort"):
        periods = min(24, n)
        schedule_data = []
        balance = L
        
        for month in range(1, periods + 1):
            interest_payment = balance * r
            principal_payment = A - interest_payment
            balance = balance - principal_payment
            
            schedule_data.append({
                "Month": month,
                "Payment": f"${A:,.2f}",
                "Interest": f"${interest_payment:,.2f}",
                "Principal": f"${principal_payment:,.2f}",
                "Remaining Balance": f"${max(balance, 0):,.2f}"
            })
        
        df = pd.DataFrame(schedule_data)
        st.dataframe(df, use_container_width=True)
        
        # Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Payment composition
        months = list(range(1, periods + 1))
        interest_payments = [float(row['Interest'].replace('$', '').replace(',', '')) for row in schedule_data]
        principal_payments = [float(row['Principal'].replace('$', '').replace(',', '')) for row in schedule_data]
        
        ax1.stackplot(months, interest_payments, principal_payments, 
                     labels=['Interest', 'Principal'], alpha=0.7)
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Payment Amount ($)')
        ax1.set_title('Payment Composition Over Time')
        ax1.legend()
        
        # Remaining balance
        balances = [float(row['Remaining Balance'].replace('$', '').replace(',', '')) for row in schedule_data]
        ax2.plot(months, balances, color='red', linewidth=2)
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Remaining Balance ($)')
        ax2.set_title('Remaining Loan Balance')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)

st.write("---")

# ---------------- QUESTION 2 ----------------
st.markdown('<div class="section-header">Question 2 — American Put Option Pricing</div>', unsafe_allow_html=True)

st.markdown("""
**Problem Statement:**  
Price a 5-month American put option on a non-dividend paying stock using a 5-step binomial tree.  
Identify nodes where early exercise is optimal. Improve the estimate using control variate technique.

**Parameters:**  
- Stock price = $50, Strike price = $50  
- Risk-free rate = 10% p.a. (continuously compounded)  
- Volatility = 40% p.a.  
- 5 time steps (1 month each)
""")

with st.expander("🔍 Show Detailed Solution & Calculator", expanded=False):
    st.markdown('<div class="subsection-header">Binomial Tree Methodology</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**Option Parameters**")
        S0 = st.number_input("Stock Price \( S_0 \)", value=50.0, min_value=0.01, step=1.0, key="q2_s0")
        K = st.number_input("Strike Price \( K \)", value=50.0, min_value=0.01, step=1.0, key="q2_k")
    
    with col2:
        st.markdown("**Market Parameters**")
        r_cont = st.number_input("Risk-free Rate (% p.a., continuous)", value=10.0, step=0.1, key="q2_r") / 100.0
        sigma = st.number_input("Volatility (% p.a.)", value=40.0, step=0.1, key="q2_sigma") / 100.0
    
    steps = 5
    T = 5 / 12  # 5 months in years
    dt = T / steps
    
    # CRR parameters with detailed calculations
    st.markdown('<div class="subsection-header">Tree Parameter Calculations</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="calculation-step">', unsafe_allow_html=True)
    st.markdown("**Step 1: Calculate Time Step**")
    st.latex(r"\Delta t = \frac{T}{n} = \frac{5/12}{5} = \frac{0.4167}{5} = 0.0833 \text{ years}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="calculation-step">', unsafe_allow_html=True)
    st.markdown("**Step 2: Calculate Up and Down Factors**")
    st.latex(r"u = e^{\sigma \sqrt{\Delta t}} = e^{0.40 \times \sqrt{0.0833}} = e^{0.40 \times 0.288675} = e^{0.11547} = 1.122401")
    st.latex(r"d = \frac{1}{u} = \frac{1}{1.122401} = 0.890947")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="calculation-step">', unsafe_allow_html=True)
    st.markdown("**Step 3: Calculate Risk-Neutral Probability**")
    st.latex(r"p = \frac{e^{r \Delta t} - d}{u - d}")
    st.latex(r"e^{r \Delta t} = e^{0.10 \times 0.0833} = e^{0.008333} = 1.008368")
    st.latex(r"p = \frac{1.008368 - 0.890947}{1.122401 - 0.890947} = \frac{0.117421}{0.231454} = 0.5073")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="calculation-step">', unsafe_allow_html=True)
    st.markdown("**Step 4: Calculate Discount Factor**")
    st.latex(r"\text{Discount} = e^{-r \Delta t} = e^{-0.10 \times 0.0833} = e^{-0.008333} = 0.991701")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Actual calculations
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp(r_cont * dt) - d) / (u - d)
    discount = math.exp(-r_cont * dt)
    
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.markdown("**Binomial Tree Parameters Summary:**")
    
    param_data = {
        'Parameter': ['Time to expiration (T)', 'Time step (Δt)', 'Up factor (u)', 'Down factor (d)', 
                     'Risk-neutral probability (p)', 'Discount factor'],
        'Value': [f"{T:.4f} years", f"{dt:.4f} years", f"{u:.6f}", f"{d:.6f}", f"{p:.6f}", f"{discount:.6f}"],
        'Formula': [
            '5/12',
            'T/n',
            r'e^{\sigma \sqrt{\Delta t}}',
            '1/u',
            r'(e^{r\Delta t} - d)/(u - d)',
            r'e^{-r\Delta t}'
        ]
    }
    
    param_df = pd.DataFrame(param_data)
    st.table(param_df)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Enhanced Binomial tree implementation with detailed tracking
    def binomial_american_put_detailed(S0, K, r, sigma, steps):
        dt = T / steps
        u = math.exp(sigma * math.sqrt(dt))
        d = 1 / u
        p = (math.exp(r * dt) - d) / (u - d)
        discount = math.exp(-r * dt)
        
        # Initialize stock prices at maturity
        stock_tree = {}
        option_tree = {}
        decision_tree = {}
        
        # Build stock price tree
        for step in range(steps + 1):
            stock_prices = []
            for i in range(step + 1):
                stock_price = S0 * (u ** (step - i)) * (d ** i)
                stock_prices.append(stock_price)
            stock_tree[step] = stock_prices
        
        # Initialize option values at maturity
        option_values = [max(K - s, 0) for s in stock_tree[steps]]
        option_tree[steps] = option_values
        decision_tree[steps] = ['Exercise' if val > 0 else 'Hold' for val in option_values]
        
        early_exercise_nodes = []
        detailed_tree_data = {steps: list(zip(stock_tree[steps], option_values))}
        
        # Backward induction with detailed tracking
        for step in range(steps - 1, -1, -1):
            new_option_values = []
            decisions = []
            
            for i in range(step + 1):
                stock_price = stock_tree[step][i]
                continuation_value = discount * (p * option_tree[step + 1][i] + (1 - p) * option_tree[step + 1][i + 1])
                exercise_value = max(K - stock_price, 0)
                option_value = max(continuation_value, exercise_value)
                
                new_option_values.append(option_value)
                
                if exercise_value > continuation_value + 1e-12:
                    decisions.append('Early Exercise')
                    early_exercise_nodes.append({
                        'Time Step': step,
                        'Node': i,
                        'Stock Price': f"${stock_price:.2f}",
                        'Exercise Value': f"${exercise_value:.4f}",
                        'Continuation Value': f"${continuation_value:.4f}",
                        'Option Value': f"${option_value:.4f}"
                    })
                else:
                    decisions.append('Hold')
            
            option_tree[step] = new_option_values
            decision_tree[step] = decisions
            detailed_tree_data[step] = list(zip(stock_tree[step], new_option_values, decisions))
        
        return option_tree[0][0], early_exercise_nodes, detailed_tree_data, stock_tree, option_tree, decision_tree
    
    # Calculate option price
    american_price, early_nodes, detailed_tree, stock_tree, option_tree, decision_tree = binomial_american_put_detailed(
        S0, K, r_cont, sigma, steps)
    
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.metric("American Put Option Price (5-step Binomial Tree)", f"${american_price:.4f}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Early exercise analysis
    if early_nodes:
        st.markdown("**🎯 Early Exercise Opportunities Detected:**")
        early_df = pd.DataFrame(early_nodes)
        st.dataframe(early_df, use_container_width=True)
    else:
        st.info("No early exercise opportunities detected in this tree.")
    
    # Enhanced Tree Visualization
    st.markdown('<div class="subsection-header">Binomial Tree Visualization</div>', unsafe_allow_html=True)
    
    # Text-based tree display
    if st.checkbox("Show Text-Based Binomial Tree", value=True, key="q2_tree_text"):
        st.markdown("**Tree Structure (Stock Price → Option Value [Decision]):**")
        
        # Create a text-based tree representation
        tree_display = []
        for step in range(steps + 1):
            level_data = []
            for i in range(len(stock_tree[step])):
                stock_val = stock_tree[step][i]
                option_val = option_tree[step][i]
                decision = decision_tree[step][i] if step < steps else "Expiry"
                
                node_info = f"${stock_val:.2f} → ${option_val:.4f} [{decision}]"
                level_data.append(node_info)
            tree_display.append(level_data)
        
        # Display tree levels
        for step, nodes in enumerate(tree_display):
            st.write(f"**Step {step}:**")
            cols = st.columns(len(nodes))
            for i, node in enumerate(nodes):
                with cols[i]:
                    node_class = "tree-node early-exercise" if (step < steps and decision_tree[step][i] == "Early Exercise") else "tree-node hold-node"
                    st.markdown(f'<div class="{node_class}">{node}</div>', unsafe_allow_html=True)
    
    # Enhanced Graphical Binomial Tree Visualization
    st.markdown('<div class="subsection-header">Enhanced Graphical Binomial Tree</div>', unsafe_allow_html=True)
    
    if st.checkbox("Show Enhanced Graphical Binomial Tree", value=True, key="q2_tree_graph"):
        
        # Create matplotlib figure for binomial tree
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Enhanced tree layout parameters
        level_height = 2.0
        node_radius = 0.25  # Larger nodes for better visibility
        font_size = 10      # Larger font for better readability
        
        # Enhanced colors with better contrast
        exercise_color = '#FF4444'  # Bright red for early exercise
        hold_color = '#44CC44'      # Bright green for hold
        expiry_color = '#4488FF'    # Bright blue for expiry
        line_color = '#888888'      # Medium gray for connections
        text_color = '#000000'      # Black text for better readability
        
        # Draw the tree
        for step in range(steps + 1):
            # Calculate x positions for this level
            node_count = step + 1
            level_width = max(3.0, (node_count - 1) * 1.5)  # Wider spacing
            x_start = -level_width / 2
            x_positions = [x_start + i * (level_width / max(1, node_count - 1)) for i in range(node_count)]
            y_position = step * level_height
            
            # Draw nodes
            for i in range(node_count):
                # Determine node color based on decision
                if step == steps:  # Final nodes
                    color = expiry_color
                    edge_color = '#003366'  # Dark blue border
                    edge_width = 2
                elif decision_tree[step][i] == "Early Exercise":
                    color = exercise_color
                    edge_color = '#990000'  # Dark red border
                    edge_width = 3
                else:
                    color = hold_color
                    edge_color = '#006600'  # Dark green border
                    edge_width = 2
                
                # Draw larger node circle with enhanced visibility
                circle = plt.Circle((x_positions[i], y_position), node_radius, 
                                  facecolor=color, edgecolor=edge_color, alpha=0.9, linewidth=edge_width)
                ax.add_patch(circle)
                
                # Add enhanced node text with better formatting
                stock_price = stock_tree[step][i]
                option_value = option_tree[step][i]
                decision = decision_tree[step][i] if step < steps else "Expiry"
                
                # Stock price (top line) - larger and bold
                ax.text(x_positions[i], y_position + 0.15, f'S: ${stock_price:.1f}', 
                       ha='center', va='center', fontsize=font_size, fontweight='bold', color=text_color)
                
                # Option value (middle line) - larger and bold
                ax.text(x_positions[i], y_position, f'O: ${option_value:.3f}', 
                       ha='center', va='center', fontsize=font_size, fontweight='bold', color=text_color)
                
                # Decision (bottom line) - larger and italic
                ax.text(x_positions[i], y_position - 0.15, f'[{decision}]', 
                       ha='center', va='center', fontsize=font_size-1, style='italic', color=text_color, fontweight='bold')
            
            # Draw enhanced connections to next level
            if step < steps:
                for i in range(node_count):
                    # Calculate positions for child nodes
                    next_node_count = step + 2
                    next_level_width = max(3.0, (next_node_count - 1) * 1.5)
                    next_x_start = -next_level_width / 2
                    next_x_positions = [next_x_start + j * (next_level_width / max(1, next_node_count - 1)) for j in range(next_node_count)]
                    
                    # Thicker connection lines
                    line_width = 1.5
                    
                    # Connection to up node
                    ax.plot([x_positions[i], next_x_positions[i]], 
                           [y_position - node_radius, (step + 1) * level_height + node_radius], 
                           color=line_color, linewidth=line_width, alpha=0.8, solid_capstyle='round')
                    
                    # Connection to down node
                    ax.plot([x_positions[i], next_x_positions[i + 1]], 
                           [y_position - node_radius, (step + 1) * level_height + node_radius], 
                           color=line_color, linewidth=line_width, alpha=0.8, solid_capstyle='round')
        
        # Enhanced plot properties
        ax.set_xlim(-steps-1, steps+1)
        ax.set_ylim(-0.5, steps * level_height + 0.5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Enhanced Binomial Tree - American Put Option\n(Stock Price → Option Value [Decision])', 
                    fontsize=16, fontweight='bold', pad=30, color='#2e86ab')
        
        # Enhanced legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=exercise_color, 
                      markersize=12, markeredgecolor='#990000', markeredgewidth=2, label='Early Exercise'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=hold_color, 
                      markersize=12, markeredgecolor='#006600', markeredgewidth=2, label='Hold'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=expiry_color, 
                      markersize=12, markeredgecolor='#003366', markeredgewidth=2, label='Expiry')
        ]
        ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), 
                 ncol=3, fontsize=12, framealpha=0.9, shadow=True)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.2, linestyle='--', color='gray')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Control Variate Technique
    st.markdown('<div class="subsection-header">Control Variate Technique</div>', unsafe_allow_html=True)
    
    st.markdown("""
    **Methodology:**  
    We use the Black-Scholes European put price as a control variate to improve the accuracy of our binomial tree estimate.
    The adjustment is based on the assumption that the error in pricing the European option is similar to the error in pricing the American option.
    """)
    
    def black_scholes_put(S, K, r, sigma, T):
        if T <= 0:
            return max(K - S, 0)
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    def binomial_european_put(S0, K, r, sigma, steps):
        dt = T / steps
        u = math.exp(sigma * math.sqrt(dt))
        d = 1.0 / u
        p = (math.exp(r * dt) - d) / (u - d)
        discount = math.exp(-r * dt)
        
        # Build stock price tree
        stock_prices = [S0 * (u ** (steps - i)) * (d ** i) for i in range(steps + 1)]
        option_values = [max(K - s, 0) for s in stock_prices]
        
        # Backward induction (European - no early exercise)
        for step in range(steps - 1, -1, -1):
            new_option_values = []
            for i in range(step + 1):
                continuation_value = discount * (p * option_values[i] + (1 - p) * option_values[i + 1])
                new_option_values.append(continuation_value)
            option_values = new_option_values
        
        return option_values[0]
    
    # Calculate European option prices
    european_bs = black_scholes_put(S0, K, r_cont, sigma, T)
    european_binomial = binomial_european_put(S0, K, r_cont, sigma, steps)
    
    # Display calculations
    st.markdown("**European Put Price Calculations:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Black-Scholes Formula:**")
        st.latex(r"P = Ke^{-rT}N(-d_2) - S_0N(-d_1)")
        st.latex(r"d_1 = \frac{\ln(S_0/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}")
        st.latex(r"d_2 = d_1 - \sigma\sqrt{T}")
        st.metric("Black-Scholes European Put", f"${european_bs:.4f}")
    
    with col2:
        st.markdown("**Binomial Tree European Put:**")
        st.metric("Binomial European Put", f"${european_binomial:.4f}")
    
    # Control variate adjustment
    adjustment = european_bs - european_binomial
    improved_american = american_price + adjustment
    
    st.markdown('<div class="calculation-step">', unsafe_allow_html=True)
    st.markdown("**Control Variate Adjustment:**")
    st.latex(r"P_{\text{American}}^{\text{improved}} = P_{\text{American}}^{\text{binomial}} + (P_{\text{European}}^{\text{BS}} - P_{\text{European}}^{\text{binomial}})")
    st.latex(rf"P_{{American}}^{{improved}} = {american_price:.4f} + ({european_bs:.4f} - {european_binomial:.4f}) = {american_price:.4f} + {adjustment:.4f}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.metric("Improved American Put Price (Control Variate)", f"${improved_american:.4f}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Summary table
    st.markdown("**Summary of Results:**")
    summary_data = {
        'Method': [
            'Binomial Tree (American)',
            'Binomial Tree (European)', 
            'Black-Scholes (European)',
            'Control Variate Adjusted (American)'
        ],
        'Price': [
            f"${american_price:.4f}",
            f"${european_binomial:.4f}",
            f"${european_bs:.4f}", 
            f"${improved_american:.4f}"
        ],
        'Notes': [
            '5-step tree with early exercise',
            '5-step tree, no early exercise',
            'Analytical solution',
            'Binomial American + (BS European - Binomial European)'
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    st.table(summary_df)

st.write("---")

# ---------------- QUESTION 3 ----------------
st.markdown('<div class="section-header">Question 3 — Efficient Portfolios (Markowitz Optimization)</div>', unsafe_allow_html=True)

st.markdown("""
**Problem Statement:**  
Construct efficient portfolios with one risk-free asset and three risky assets.  
Expected returns: 6%, 10%, 12%, 18% respectively.  
Covariance matrix for risky assets:
""")
st.latex(r"""
C = \begin{bmatrix}
4 & 20 & 40 \\
20 & 10 & 70 \\
40 & 70 & 14
\end{bmatrix}
""")

with st.expander("🔍 Show Detailed Solution & Calculator", expanded=False):
    st.markdown('<div class="subsection-header">Mathematical Framework</div>', unsafe_allow_html=True)
    
    st.markdown("""
    **Key Concepts:**
    - When a risk-free asset exists, all efficient portfolios lie on the **Capital Market Line (CML)**
    - The **Tangency Portfolio** is the optimal risky portfolio that maximizes the Sharpe ratio
    - Any efficient portfolio is a combination of the risk-free asset and the tangency portfolio
    """)
    
    st.markdown("**Tangency Portfolio Weights:**")
    st.latex(r"w_{\text{tan}} \propto \Sigma^{-1} (\mu - r_f \mathbf{1})")
    st.markdown('<div class="annotation">Σ is the covariance matrix, μ is the vector of expected returns, r_f is the risk-free rate</div>', unsafe_allow_html=True)
    
    # Input parameters
    st.markdown('<div class="subsection-header">Portfolio Parameters</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Expected Returns (%)**")
        rf = st.number_input("Risk-free rate (a₁)", value=6.0, step=0.1, key="q3_rf") / 100.0
        r2 = st.number_input("Asset a₂", value=10.0, step=0.1, key="q3_r2") / 100.0
        r3 = st.number_input("Asset a₃", value=12.0, step=0.1, key="q3_r3") / 100.0
        r4 = st.number_input("Asset a₄", value=18.0, step=0.1, key="q3_r4") / 100.0
    
    with col2:
        st.markdown("**Covariance Matrix**")
        C = np.array([
            [4.0, 20.0, 40.0],
            [20.0, 10.0, 70.0],
            [40.0, 70.0, 14.0]
        ])
        st.dataframe(pd.DataFrame(C, index=['a₂', 'a₃', 'a₄'], columns=['a₂', 'a₃', 'a₄']))
    
    # Calculations
    mu_risky = np.array([r2, r3, r4])
    excess_returns = mu_risky - rf
    
    try:
        # Tangency portfolio calculation
        C_inv = np.linalg.inv(C)
        w_unscaled = C_inv @ excess_returns
        w_tangency = w_unscaled / np.sum(w_unscaled)
        
        # Portfolio characteristics
        mu_tangency = w_tangency @ mu_risky
        var_tangency = w_tangency @ C @ w_tangency
        sd_tangency = np.sqrt(var_tangency)
        sharpe_ratio = (mu_tangency - rf) / sd_tangency
        
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.markdown("**Tangency Portfolio Results**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Expected Return", f"{mu_tangency*100:.2f}%")
        with col2:
            st.metric("Standard Deviation", f"{sd_tangency*100:.2f}%")
        with col3:
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.4f}")
        
        st.markdown("**Portfolio Weights:**")
        weights_df = pd.DataFrame({
            'Asset': ['a₂', 'a₃', 'a₄'],
            'Weight': w_tangency,
            'Percentage': [f"{w*100:.2f}%" for w in w_tangency]
        })
        st.dataframe(weights_df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Efficient frontier calculator
        st.markdown('<div class="subsection-header">Efficient Portfolio Calculator</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            weight_tangency = st.slider("Weight in Tangency Portfolio", -0.5, 2.0, 1.0, 0.01, key="q3_weight")
            weight_rf = 1 - weight_tangency
            
            if weight_tangency > 1:
                st.warning(f"Leveraged portfolio: Borrowing {((weight_tangency-1)*100):.1f}% at risk-free rate")
            elif weight_tangency < 0:
                st.warning(f"Short selling tangency portfolio: {(-weight_tangency*100):.1f}%")
            else:
                st.success(f"Standard portfolio: {weight_tangency*100:.1f}% tangency, {weight_rf*100:.1f}% risk-free")
        
        with col2:
            portfolio_return = rf + weight_tangency * (mu_tangency - rf)
            portfolio_std = abs(weight_tangency) * sd_tangency
            
            st.metric("Portfolio Expected Return", f"{portfolio_return*100:.2f}%")
            st.metric("Portfolio Standard Deviation", f"{portfolio_std*100:.2f}%")
        
        # CML visualization
        st.markdown('<div class="subsection-header">Capital Market Line</div>', unsafe_allow_html=True)
        
        # Generate CML points
        sigmas = np.linspace(0, sd_tangency * 2.5, 100)
        returns = rf + sharpe_ratio * sigmas
        
        fig = go.Figure()
        
        # CML line
        fig.add_trace(go.Scatter(
            x=sigmas*100, y=returns*100,
            mode='lines',
            name='Capital Market Line',
            line=dict(color='blue', width=3)
        ))
        
        # Tangency portfolio
        fig.add_trace(go.Scatter(
            x=[sd_tangency*100], y=[mu_tangency*100],
            mode='markers',
            name='Tangency Portfolio',
            marker=dict(color='red', size=12, symbol='star')
        ))
        
        # Risk-free asset
        fig.add_trace(go.Scatter(
            x=[0], y=[rf*100],
            mode='markers',
            name='Risk-Free Asset',
            marker=dict(color='green', size=10)
        ))
        
        # Current portfolio
        fig.add_trace(go.Scatter(
            x=[portfolio_std*100], y=[portfolio_return*100],
            mode='markers',
            name='Your Portfolio',
            marker=dict(color='orange', size=10, symbol='diamond')
        ))
        
        fig.update_layout(
            title='Capital Market Line (CML)',
            xaxis_title='Portfolio Standard Deviation (%)',
            yaxis_title='Portfolio Expected Return (%)',
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except np.linalg.LinAlgError:
        st.error("Covariance matrix is singular. Cannot compute efficient portfolios.")

st.write("---")

# ---------------- QUESTION 4 ----------------
st.markdown('<div class="section-header">Question 4 — Capital Market Line Applications</div>', unsafe_allow_html=True)

st.markdown("""
**Problem Statement:**  
Given market portfolio expected return = 23%, risk-free rate = 7%, market standard deviation = 32%.

**(a)** Find the equation of the Capital Market Line (CML)  
**(b)** Calculate expected return for a portfolio with Rs 300 in risk-free asset and Rs 700 in market portfolio  
**(c)** Determine portfolio composition to achieve 39% return with Rs 1000 investment
""")

with st.expander("🔍 Show Detailed Solution & Calculator", expanded=False):
    st.markdown('<div class="subsection-header">Market Parameters</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Rm = st.number_input("Market Return (% p.a.)", value=23.0, step=0.1, key="q4_Rm") / 100.0
    with col2:
        Rf = st.number_input("Risk-Free Rate (% p.a.)", value=7.0, step=0.1, key="q4_Rf") / 100.0
    with col3:
        sigma_m = st.number_input("Market Std Dev (% p.a.)", value=32.0, step=0.1, key="q4_sigma") / 100.0
    
    # CML calculation
    sharpe_ratio = (Rm - Rf) / sigma_m
    
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.markdown("**(a) Capital Market Line Equation**")
    st.latex(r"E[R_p] = R_f + \frac{E[R_m] - R_f}{\sigma_m} \times \sigma_p")
    st.write(f"**Numerical Form:** \( E[R_p] = {Rf:.4f} + {sharpe_ratio:.4f} \\times \\sigma_p \)")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Part (b)
    st.markdown('<div class="subsection-header">(b) Specific Portfolio Return</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        invest_rf = st.number_input("Investment in Risk-Free (Rs)", value=300.0, step=10.0, key="q4_inv_rf")
        invest_market = st.number_input("Investment in Market (Rs)", value=700.0, step=10.0, key="q4_inv_mkt")
        total_investment = invest_rf + invest_market
        
        w_rf = invest_rf / total_investment
        w_market = invest_market / total_investment
        portfolio_return_b = w_rf * Rf + w_market * Rm
        
        st.metric("Portfolio Expected Return", f"{portfolio_return_b*100:.2f}%")
    
    with col2:
        st.markdown('<div class="calculation-step">', unsafe_allow_html=True)
        st.write("**Calculation:**")
        st.write(f"Weight in risk-free: \( {invest_rf} / {total_investment} = {w_rf:.4f} \)")
        st.write(f"Weight in market: \( {invest_market} / {total_investment} = {w_market:.4f} \)")
        st.write(f"\( E[R_p] = {w_rf:.4f} \\times {Rf:.4f} + {w_market:.4f} \\times {Rm:.4f} = {portfolio_return_b:.4f} \)")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Part (c)
    st.markdown('<div class="subsection-header">(c) Target Return Portfolio</div>', unsafe_allow_html=True)
    
    target_return = st.number_input("Target Return (% p.a.)", value=39.0, step=0.1, key="q4_target") / 100.0
    total_capital = 1000.0
    
    if abs(Rm - Rf) > 1e-10:
        w_market_req = (target_return - Rf) / (Rm - Rf)
        w_rf_req = 1 - w_market_req
        
        invest_market_req = w_market_req * total_capital
        invest_rf_req = w_rf_req * total_capital
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="calculation-step">', unsafe_allow_html=True)
            st.markdown("**Required Portfolio Weights:**")
            st.write(f"Weight in market portfolio: \( \\frac{{{target_return:.4f} - {Rf:.4f}}}{{{Rm:.4f} - {Rf:.4f}}} = {w_market_req:.4f} \)")
            st.write(f"Weight in risk-free asset: \( 1 - {w_market_req:.4f} = {w_rf_req:.4f} \)")
            
            if w_market_req > 1:
                st.warning(f"**Strategy:** Borrow Rs {abs(invest_rf_req):.2f} at risk-free rate and invest Rs {invest_market_req:.2f} in market portfolio")
            elif w_market_req < 0:
                st.warning(f"**Strategy:** Short sell market portfolio for Rs {abs(invest_market_req):.2f} and invest Rs {invest_rf_req:.2f} in risk-free asset")
            else:
                st.success(f"**Strategy:** Invest Rs {invest_market_req:.2f} in market portfolio and Rs {invest_rf_req:.2f} in risk-free asset")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="calculation-step">', unsafe_allow_html=True)
            st.markdown("**Verification:**")
            verified_return = w_rf_req * Rf + w_market_req * Rm
            st.write(f"\( {w_rf_req:.4f} \\times {Rf:.4f} + {w_market_req:.4f} \\times {Rm:.4f} = {verified_return:.4f} \)")
            st.success(f"Target return achieved: {verified_return*100:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        st.error("Market return equals risk-free rate. Cannot form leveraged portfolio.")

st.write("---")

# Footer
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    Financial Mathematics Assignment Solutions • Interactive Calculator • Created with Streamlit
</div>
""", unsafe_allow_html=True)
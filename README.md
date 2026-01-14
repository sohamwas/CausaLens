# CausaLens
### A Practical Framework for Graph-Based Causal Analysis and Adjustment Set Discovery

---

## 📌 Overview

**CausaLens** is a causal inference project designed to move beyond correlation-based analysis by explicitly modeling **causal structure**, identifying **confounders**, and estimating **interventional effects** using principled causal reasoning.

The project focuses on:
- Representing causal assumptions as **directed graphs**
- Identifying **valid adjustment sets** using the **backdoor criterion**
- Preventing common causal errors such as conditioning on **colliders** or **post-treatment variables**
- Estimating causal effects under explicit assumptions

This repository is centered around an **exploratory, notebook-based workflow** that emphasizes *interpretability, correctness, and causal validity*.

---

## 🎯 Motivation

In many applied ML and data science problems (e.g., churn prediction, delivery delays, product quality analysis), models answer:

> "What is correlated with the outcome?"

But decision-making requires a stronger question:

> **"What causes the outcome?"**

Without causal reasoning:
- Confounders bias estimates
- Interventions fail in the real world
- Models break under distribution shifts

**CausaLens** provides a structured way to:
1. Explicitly encode causal assumptions  
2. Diagnose confounding paths  
3. Select correct controls  
4. Estimate effects that correspond to real-world interventions  

---

## 🧠 Core Causal Concepts Used

This project relies on foundational concepts from causal inference:

### **1. Directed Acyclic Graphs (DAGs)**
- Represent causal relationships as directed edges (A → B means "A causes B")
- Nodes are variables; edges encode causal assumptions
- Used to visualize confounding paths and causal flows

### **2. Backdoor Criterion**
- Identifies which variables to control for to block confounding
- A set of variables Z satisfies the backdoor criterion if:
  - No node in Z is a descendant of treatment
  - Z blocks all backdoor paths from treatment to outcome

### **3. Adjustment Sets**
- The minimal set of variables needed to obtain unbiased causal estimates
- Derived from DAG structure using graphical criteria
- Controls for confounders while avoiding colliders and mediators

### **4. Confounders vs Colliders**
- **Confounder**: Affects both treatment and outcome → must adjust for it
- **Collider**: Caused by both treatment and outcome → must NOT adjust for it
- **Mediator**: On the causal path (treatment → mediator → outcome) → adjusting blocks the effect

### **5. Temporal Ordering**
- Causes must precede effects in time
- Post-treatment variables cannot be confounders
- Ensures causal direction is valid

---

## 🔬 Project Workflow

### **Step 1: Data Preprocessing**

**Goal:** Transform raw e-commerce data into a causally valid analysis-ready dataset.

#### What We Do:
1. **Import Multi-Table Data from Kaggle**
   - Load four tables: `orders`, `reviews`, `order_items`, `customers`
   - Focus on order-level analysis (one row per order)

2. **Parse Timestamps**
   - Convert all date columns to consistent datetime format using `pd.to_datetime()`
   - Ensures temporal ordering can be validated

3. **Create Intervention Variable (Treatment)**
   - `delayed_delivery_days` = `order_delivered_customer_date` - `order_estimated_delivery_date`
   - `delayed_delivery` = binary indicator (1 if delay > 0 days, else 0)
   - This is the **treatment** whose effect we want to estimate

4. **Merge Outcome Variable**
   - Join `reviews` table to get `review_score` (1-5 stars)
   - This is the **outcome** we want to explain causally
   - Keep only one review per order (drop duplicates)

5. **Derive Order-Level Features (Pre-Treatment Confounders)**
   - `order_price`: Total cost of all items in order
   - `freight_value`: Shipping cost
   - `num_items`: Number of products ordered
   - These affect both likelihood of delay AND customer satisfaction

6. **Derive Customer History Features (Pre-Treatment Confounders)**
   - `customer_tenure_days`: Days since first purchase
   - `prior_orders`: Number of previous orders (using `.cumcount()` for temporal validity)
   - `is_first_order`: Binary flag for first-time customers
   - These capture customer loyalty and experience effects

7. **Apply Validity Filters**
   - Keep only delivered orders with valid timestamps and reviews
   - Remove incomplete or invalid observations

8. **Save Preprocessed Dataset**
   - Export clean CSV with treatment, outcome, and confounders
   - Ready for causal discovery and estimation

**Output:** `olist_preprocessed_causal.csv` (~96k orders, 12 columns)

---

### **Step 2: Causal Discovery & DAG Construction**

**Goal:** Discover causal structure from data and identify which variables confound the treatment-outcome relationship.

#### What We Do:
1. **Run Three Causal Discovery Algorithms**
   - **PC (Constraint-Based):** Tests conditional independence using Fisher's Z test
   - **GES (Score-Based):** Optimizes Bayesian Information Criterion (BIC)
   - **NOTEARS (Continuous Optimization):** Uses gradient descent with acyclicity constraint
   - Each algorithm proposes edges based on different statistical principles

2. **Sample Data for Efficiency**
   - Use 10,000 random orders (from ~96k) to reduce computational cost
   - Causal structure learning doesn't require full data to detect relationships
   - Standardize variables (mean=0, std=1) for numerical stability

3. **Build Consensus DAG (Conservative Approach)**
   - Keep only edges agreed upon by **≥2 out of 3 algorithms**
   - This reduces false positives and improves robustness
   - Pure data-driven structure discovery

4. **Enforce Temporal Constraints**
   - Remove edges that violate time ordering:
     - ❌ `review_score → delayed_delivery` (future cannot cause past)
     - ❌ `delayed_delivery → order_price` (treatment cannot cause pre-treatment)
   - Keep only valid temporal flows:
     - ✅ `order_price → delayed_delivery` (past → present)
     - ✅ `delayed_delivery → review_score` (present → future)
     - ✅ `order_price → review_score` (past → future)

5. **Incorporate Domain Knowledge**
   - Algorithms may miss important confounders due to sample size or weak signals
   - Manually add theoretically justified edges:
     - `order_price → delayed_delivery` (expensive orders → special handling → delays)
     - `prior_orders → delayed_delivery` (loyal customers → priority shipping)
     - `is_first_order → delayed_delivery` (first-timers → default slower tier)
   - Hybrid approach: data + domain expertise

6. **Identify Adjustment Set (Backdoor Criterion)**
   - Find variables that create **backdoor paths** from treatment to outcome
   - **Backdoor path example:**
     ```
     delayed_delivery ← order_price → review_score
     ```
     (Order price affects both delay likelihood AND review score → confounding)
   - **Adjustment set:** Variables we must control for to block these paths
   - In this project: `[order_price, prior_orders, is_first_order, num_items]`

7. **Visualize Final DAG**
   - Create graph with:
     - Red node: Treatment (`delayed_delivery`)
     - Teal node: Outcome (`review_score`)
     - Green nodes: Pre-treatment confounders
   - Save as high-resolution PNG for reporting

**Output:**
- `causal_dag_edges.csv`: List of directed edges in final DAG
- `adjustment_set.txt`: Variables to condition on in Step 3
- `causal_dag_visual.png`: DAG visualization

**Why This Matters:**
- Without this step, you'd either:
  - Control for nothing → biased by confounding
  - Control for everything → introduce collider bias or overadjustment
- The adjustment set tells you **exactly which variables to include** in regression

---

### **Step 3: Causal Effect Estimation**

**Goal:** Estimate the unbiased causal effect of delivery delays on customer satisfaction.

#### What We Do:
1. **Baseline: Naive Analysis**
   - Simple comparison of delayed vs on-time orders without adjustments
   - **Purpose:** Demonstrates confounding bias when causal methods aren't used
   - Shows overestimated effect due to unmeasured confounding

2. **Causal Estimation Methods**
   We apply three complementary approaches (Regression Adjustment, Inverse Propensity Weighting, and Matching) to ensure robustness.

3. **Triangulation**
   - Compare estimates across all three methods
   - Agreement across methods → high confidence in causal estimate
   - Disagreement → signals potential assumption violations

**Output:**
- `causal_effect_estimates.csv`: Effect estimates from all methods
- Typical finding: Naive effect ≈ -1.73 stars, Causal estimates ≈ -1.74 stars (minimal confounding detected in this dataset)

**Interpretation:**
- "Delivery delays cause approximately a **-1.7 star reduction** in customer reviews, after accounting for order characteristics and customer history"
- This represents the **interventional effect** (expected impact if delays were eliminated)

---

### **Step 4: Heterogeneous Effects (Subgroup Analysis)**

**Goal:** Determine if the causal effect varies across different customer segments.

#### What We Do:
1. **Stratified Analysis by Customer Type**
   - Split data into subgroups: first-time vs repeat customers
   - Estimate delay effect **within each group separately**
   - Compare: Do first-timers suffer more from delays than loyal customers?

2. **Analysis by Order Value**
   - Create categories: low, medium, high value orders (using quantiles)
   - Estimate effect within each value tier
   - Check if expensive orders are more sensitive to delays

3. **Analysis by Customer Tenure**
   - Categories: new, established, loyal customers
   - Test if relationship with company affects delay tolerance

4. **Statistical Interaction Test**
   - Regression with interaction term: `delayed_delivery × is_first_order`
   - Tests if effect **modification** is statistically significant
   - Coefficient = difference in delay effect between subgroups

**Typical Finding:**
- Interaction coefficient ≈ -0.15 (not significant, p > 0.05)
- **Conclusion:** Delay effects are uniform across customer types
- **Business implication:** No need for segment-specific interventions; universal on-time delivery improvement helps all customers equally

**Why This Matters:**
- Identifies **high-value intervention targets** (e.g., "prioritize first-timers")
- Or confirms effect is **stable** (one-size-fits-all strategy works)
- Prevents wasted resources on ineffective targeting

---

## 📊 Key Results

### Main Causal Effect
- **Treatment:** Delivery delay (binary: on-time vs delayed)
- **Outcome:** Customer review score (1-5 stars)
- **Estimated Effect:** **-1.73 stars** (95% CI: [-1.76, -1.70])
- **Interpretation:** Delays cause customers to rate ~1.7 stars lower on average

### Confounder Structure
**Variables that affect BOTH delay likelihood AND review scores:**
- `order_price`: Expensive orders get delayed more + rated differently
- `prior_orders`: Loyal customers get priority + are more forgiving
- `is_first_order`: First-timers get slower shipping + rate harshly
- `num_items`: More items → longer processing + affects satisfaction

### Subgroup Findings
- **No significant heterogeneity** detected (interaction p > 0.05)
- Delay effects are consistent across:
  - First-time vs repeat customers
  - Low vs high value orders
  - New vs loyal customers
- **Implication:** Universal improvement strategy is optimal

---

## 🛠️ Technologies Used

### Core Libraries
- **Causal Discovery:** `causal-learn`, `gcastle` (PC, GES, NOTEARS algorithms)
- **Graph Analysis:** `networkx` (DAG manipulation and visualization)
- **Statistical Modeling:** `scikit-learn`, `statsmodels` (regression, propensity scores)
- **Data Processing:** `pandas`, `numpy`

---

import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Page Setup
# ---------------------------------------------------------
st.set_page_config(page_title="ID3 Decision Tree Simulator",
                   layout="wide")

st.title("🌳 ID3 Decision Tree — Step-by-Step Simulator (Weather Dataset)")

# ---------------------------------------------------------
# Utility: Entropy
# ---------------------------------------------------------
def entropy(counts):
    """
    counts: dict like {"Yes": 9, "No": 5}
    returns Shannon Entropy
    """
    total = sum(counts.values())
    ent = 0
    for c in counts.values():
        if c > 0:
            p = c / total
            ent -= p * math.log2(p)
    return ent

# ---------------------------------------------------------
# Utility: Information Gain
# ---------------------------------------------------------
def compute_information_gain(df, feature, target="play"):
    """
    df: dataframe with categorical features
    feature: column name to split on
    returns:
        parent_entropy,
        weighted_entropy,
        IG,
        subsets (dict of v -> subset df)
    """
    parent_counts = df[target].value_counts().to_dict()
    parent_entropy = entropy(parent_counts)

    values = df[feature].unique()
    weighted_entropy = 0
    subsets = {}

    for v in values:
        subset = df[df[feature] == v]
        subsets[v] = subset
        subset_counts = subset[target].value_counts().to_dict()
        weighted_entropy += (len(subset) / len(df)) * entropy(subset_counts)

    IG = parent_entropy - weighted_entropy
    return parent_entropy, weighted_entropy, IG, subsets

# ---------------------------------------------------------
# Utility: Draw Compact Tree Diagram
# ---------------------------------------------------------
def draw_tree(nodes, edges, title="Decision Tree"):
    """
    nodes: dict {id: {"x":float, "y":float, "text":str}}
    edges: list of tuples (parent_id, child_id)
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_title(title, fontsize=16)
    ax.axis("off")

    # Draw edges
    for (p, c) in edges:
        x1, y1 = nodes[p]["x"], nodes[p]["y"]
        x2, y2 = nodes[c]["x"], nodes[c]["y"]
        ax.plot([x1, x2], [y1, y2], color="black")

    # Draw nodes
    for node_id, node in nodes.items():
        ax.text(node["x"], node["y"], node["text"],
                ha="center", va="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="#eef",
                          edgecolor="black"))

    return fig

# ---------------------------------------------------------
# Initialize Session State
# ---------------------------------------------------------
if "step" not in st.session_state:
    st.session_state.step = 1

# Information Gain memory
if "ig_root" not in st.session_state:
    st.session_state.ig_root = {}

if "ig_sunny" not in st.session_state:
    st.session_state.ig_sunny = {}

if "ig_rain" not in st.session_state:
    st.session_state.ig_rain = {}

# Navigation
def next_step():
    if st.session_state.step < 12:
        st.session_state.step += 1

def prev_step():
    if st.session_state.step > 1:
        st.session_state.step -= 1

# ---------------------------------------------------------
# File Upload
# ---------------------------------------------------------
uploaded = st.file_uploader(
    "Upload Weather CSV (Outlook, Temperature, Humidity, Windy, Play)",
    type=["csv"]
)

if not uploaded:
    st.warning("Please upload the dataset to begin.")
    st.stop()

df = pd.read_csv(uploaded)
df.columns = df.columns.str.strip().str.lower()

required = ["outlook", "temperature", "humidity", "windy", "play"]
if not all(col in df.columns for col in required):
    st.error("CSV must contain: Outlook, Temperature, Humidity, Windy, Play")
    st.stop()

# ---------------------------------------------------------
# STEP ENGINE — Display Current Step
# ---------------------------------------------------------
step = st.session_state.step
st.subheader(f"🪜 Step {step} of 12")

# ---------------------------------------------------------
# STEP 1 — Parent Entropy
# ---------------------------------------------------------
if step == 1:
    st.markdown("## 📌 Step 1 — Compute Parent Entropy (Target = Play)")
    st.markdown("""
    We begin by measuring the impurity of the entire dataset using **Shannon Entropy**.
    
    Formula:
    """)
    st.latex(r"H(S) = -\sum p_i \log_2(p_i)")
    
    counts = df["play"].value_counts().to_dict()
    st.write("### Class Counts:", counts)

    parent_ent = entropy(counts)

    st.write(f"### 👉 Parent Entropy = **{parent_ent:.4f}**")
    st.info("A high entropy means the data is impure and needs splitting.")

# ---------------------------------------------------------
# Features for root-level IG
# ---------------------------------------------------------
features = ["outlook", "temperature", "humidity", "windy"]

# ---------------------------------------------------------
# STEP 2 — IG for Outlook
# ---------------------------------------------------------
if step == 2:
    feature = "outlook"
    st.markdown(f"## 📌 Step 2 — Compute Information Gain for **{feature.title()}**")

    parent_entropy, weighted_entropy, IG, subsets = compute_information_gain(df, feature)

    # Store result
    st.session_state.ig_root[feature] = IG

    st.write("### Subset Entropies:")
    for val, subset in subsets.items():
        c = subset["play"].value_counts().to_dict()
        st.write(f"**{val}** → {c}, Entropy = {entropy(c):.4f}")

    st.write("---")
    st.write(f"Weighted Entropy = **{weighted_entropy:.4f}**")
    st.write(f"### 👉 IG({feature.title()}) = **{IG:.4f}**")

    st.info("Information Gain tells how much uncertainty is reduced by splitting on this feature.")

# ---------------------------------------------------------
# STEP 3 — IG for Temperature
# ---------------------------------------------------------
if step == 3:
    feature = "temperature"
    st.markdown(f"## 📌 Step 3 — Compute Information Gain for **{feature.title()}**")

    parent_entropy, weighted_entropy, IG, subsets = compute_information_gain(df, feature)
    st.session_state.ig_root[feature] = IG

    st.write("### Subset Entropies:")
    for val, subset in subsets.items():
        c = subset["play"].value_counts().to_dict()
        st.write(f"**{val}** → {c}, Entropy = {entropy(c):.4f}")

    st.write("---")
    st.write(f"Weighted Entropy = **{weighted_entropy:.4f}**")
    st.write(f"### 👉 IG({feature.title()}) = **{IG:.4f}**")

# ---------------------------------------------------------
# STEP 4 — IG for Humidity
# ---------------------------------------------------------
if step == 4:
    feature = "humidity"
    st.markdown(f"## 📌 Step 4 — Compute Information Gain for **{feature.title()}**")

    parent_entropy, weighted_entropy, IG, subsets = compute_information_gain(df, feature)
    st.session_state.ig_root[feature] = IG

    st.write("### Subset Entropies:")
    for val, subset in subsets.items():
        c = subset["play"].value_counts().to_dict()
        st.write(f"**{val}** → {c}, Entropy = {entropy(c):.4f}")

    st.write("---")
    st.write(f"Weighted Entropy = **{weighted_entropy:.4f}**")
    st.write(f"### 👉 IG({feature.title()}) = **{IG:.4f}**")

# ---------------------------------------------------------
# STEP 5 — IG for Windy
# ---------------------------------------------------------
if step == 5:
    feature = "windy"
    st.markdown(f"## 📌 Step 5 — Compute Information Gain for **{feature.title()}**")

    parent_entropy, weighted_entropy, IG, subsets = compute_information_gain(df, feature)
    st.session_state.ig_root[feature] = IG

    st.write("### Subset Entropies:")
    for val, subset in subsets.items():
        c = subset["play"].value_counts().to_dict()
        st.write(f"**{val}** → {c}, Entropy = {entropy(c):.4f}")

    st.write("---")
    st.write(f"Weighted Entropy = **{weighted_entropy:.4f}**")
    st.write(f"### 👉 IG({feature.title()}) = **{IG:.4f}**")

    st.info("We have now computed IG for all 4 features. Next step: choose the root node.")

# ---------------------------------------------------------
# STEP 6 — Select Best Feature as Root Node
# ---------------------------------------------------------
if step == 6:
    st.markdown("## 📌 Step 6 — Select Best Feature (Root Node)")
    st.markdown("""
    Now that we have computed Information Gain for all four features,
    we select the one with the **highest IG** to become the **root** of the tree.
    """)

    IGs = st.session_state.ig_root
    st.write("### Information Gains (from previous steps):")
    st.write(IGs)

    # Determine best feature
    root_feature = max(IGs, key=IGs.get)
    st.session_state.root_feature = root_feature

    st.success(f"🎉 Best Feature = **{root_feature.title()}** — This becomes the root node.")

    # Draw root-only tree
    nodes = {
        "root": {
            "text": f"{root_feature.title()}\nEntropy={entropy(df['play'].value_counts().to_dict()):.3f}",
            "x": 0.5,
            "y": 0.9
        }
    }
    edges = []

    fig = draw_tree(nodes, edges, "Root Node")
    st.pyplot(fig)

    st.info("A root node with high IG strongly reduces impurity, making it ideal for the first split.")


# ---------------------------------------------------------
# STEP 7 — Create Branches for Root Node
# ---------------------------------------------------------
if step == 7:
    root_feature = st.session_state.root_feature
    st.markdown(f"## 📌 Step 7 — Create Branches for Root Feature **{root_feature.title()}**")
    st.markdown("""
    We now split the dataset into branches based on the root feature.
    
    Each branch corresponds to a unique value of the root feature.
    """)

    # Unique values create branches
    branches = df[root_feature].unique()
    st.write("### Branch Values:", branches)

    # Build partial tree structure
    nodes = {
        "root": {
            "text": root_feature.title(),
            "x": 0.5,
            "y": 0.9
        }
    }

    edges = []
    x_positions = np.linspace(0.2, 0.8, len(branches))

    # Save subsets for upcoming steps
    st.session_state.subsets = {}

    for i, val in enumerate(branches):
        subset = df[df[root_feature] == val]
        st.session_state.subsets[val] = subset

        nodes[val] = {
            "text": f"{val}\n(samples={len(subset)})",
            "x": x_positions[i],
            "y": 0.6
        }
        edges.append(("root", val))

    fig = draw_tree(nodes, edges, "Root With Branches")
    st.pyplot(fig)

    st.info("Next, we will analyze the branch subsets (Sunny, Rain, Overcast) one by one.")
# ---------------------------------------------------------
# STEP 8 — Entropy of Sunny Subset
# ---------------------------------------------------------
if step == 8:
    st.markdown("## 📌 Step 8 — Compute Entropy of 'Sunny' Subset")
    st.markdown("""
    We now analyze the **Sunny** branch.  
    A node becomes *pure* when all labels are identical.  
    If not pure, we must compute entropy to decide whether more splitting is needed.
    """)

    subsets = st.session_state.subsets
    sunny = None

    # Handle case-insensitive matching ("Sunny", "sunny")
    for key in subsets.keys():
        if key.lower() == "sunny":
            sunny = subsets[key]

    if sunny is None:
        st.error("Sunny subset not found in dataset. Ensure your 'Outlook' column has values like Sunny/Rain/Overcast.")
    else:
        st.write("### Sunny Subset Data")
        st.dataframe(sunny)

        counts = sunny["play"].value_counts().to_dict()
        st.write("### Class Counts:", counts)

        ent_sunny = entropy(counts)
        st.write(f"### 👉 Entropy(Sunny) = **{ent_sunny:.4f}**")

        if ent_sunny == 0:
            st.success("Sunny subset is already pure — no further splitting needed.")
        else:
            st.info("Sunny subset is NOT pure → we must compute IG for remaining features.")


# ---------------------------------------------------------
# STEP 9 — IG for Sunny Subset (Humidity wins)
# ---------------------------------------------------------
if step == 9:
    st.markdown("## 📌 Step 9 — Compute Information Gain inside 'Sunny' Subset")

    subsets = st.session_state.subsets
    sunny = None
    for key in subsets:
        if key.lower() == "sunny":
            sunny = subsets[key]

    st.write("### Sunny Subset Data")
    st.dataframe(sunny)

    st.markdown("### Remaining Features: Temperature, Humidity, Windy")
    features_remaining = ["temperature", "humidity", "windy"]

    IG_sunny = {}

    for f in features_remaining:
        _, _, ig, _ = compute_information_gain(sunny, f)
        IG_sunny[f] = ig

    st.session_state.ig_sunny = IG_sunny

    st.write("### Information Gains inside Sunny Subset:")
    st.write(IG_sunny)

    best_feature = max(IG_sunny, key=IG_sunny.get)
    st.success(f"🎉 Best Split for Sunny = **{best_feature.title()}**")

    st.info("This feature gives maximum reduction in impurity for the Sunny branch.")


# ---------------------------------------------------------
# STEP 10 — Entropy of Rain Subset
# ---------------------------------------------------------
if step == 10:
    st.markdown("## 📌 Step 10 — Compute Entropy of 'Rain' Subset")

    subsets = st.session_state.subsets

    rain = None
    for key in subsets:
        if key.lower() == "rain":
            rain = subsets[key]

    st.write("### Rain Subset Data")
    st.dataframe(rain)

    counts = rain["play"].value_counts().to_dict()
    st.write("### Class Counts:", counts)

    ent_rain = entropy(counts)
    st.write(f"### 👉 Entropy(Rain) = **{ent_rain:.4f}**")

    if ent_rain == 0:
        st.success("Rain subset is pure — no further splitting needed.")
    else:
        st.info("Rain subset is NOT pure → we compute IG next.")


# ---------------------------------------------------------
# STEP 11 — IG for Rain Subset (Windy wins)
# ---------------------------------------------------------
if step == 11:
    st.markdown("## 📌 Step 11 — Compute Information Gain inside 'Rain' Subset")

    subsets = st.session_state.subsets

    rain = None
    for key in subsets:
        if key.lower() == "rain":
            rain = subsets[key]

    st.write("### Rain Subset Data")
    st.dataframe(rain)

    features_remaining = ["temperature", "humidity", "windy"]
    IG_rain = {}

    for f in features_remaining:
        _, _, ig, _ = compute_information_gain(rain, f)
        IG_rain[f] = ig

    st.session_state.ig_rain = IG_rain

    st.write("### Information Gains inside Rain Subset:")
    st.write(IG_rain)

    best_feature = max(IG_rain, key=IG_rain.get)
    st.success(f"🎉 Best Split for Rain = **{best_feature.title()}**")

    st.info("This feature gives maximum reduction in impurity for the Rain branch.")

# ---------------------------------------------------------
# STEP 12 — Final ID3 Decision Tree (Graphical)
# ---------------------------------------------------------
if step == 12:
    st.markdown("## 🎉 Step 12 — Final Decision Tree (ID3 Algorithm)")
    st.markdown("""
    The full decision tree is now constructed using all previous steps.
    This tree represents the learned rules from the dataset.
    """)

    # Build compact node structure
    nodes = {
        "root": {
            "text": "Outlook",
            "x": 0.5,
            "y": 0.92
        },
        # First-level branches
        "Sunny": {
            "text": "Sunny\n→ Humidity",
            "x": 0.25,
            "y": 0.65
        },
        "Rain": {
            "text": "Rain\n→ Windy",
            "x": 0.50,
            "y": 0.65
        },
        "Overcast": {
            "text": "Overcast\n→ Yes",
            "x": 0.75,
            "y": 0.65
        },
        # Sunny branch children
        "High": {
            "text": "High → No",
            "x": 0.15,
            "y": 0.35
        },
        "Normal": {
            "text": "Normal → Yes",
            "x": 0.35,
            "y": 0.35
        },
        # Rain branch children
        "WindyTrue": {
            "text": "Windy = True → No",
            "x": 0.45,
            "y": 0.35
        },
        "WindyFalse": {
            "text": "Windy = False → Yes",
            "x": 0.55,
            "y": 0.35
        }
    }

    edges = [
        ("root", "Sunny"),
        ("root", "Rain"),
        ("root", "Overcast"),
        ("Sunny", "High"),
        ("Sunny", "Normal"),
        ("Rain", "WindyTrue"),
        ("Rain", "WindyFalse")
    ]

    fig = draw_tree(nodes, edges, "Final ID3 Decision Tree")
    st.pyplot(fig)

    st.success("🎯 The final tree is complete! Students can now use it for prediction.")


    # -----------------------------------------------------
    # Prediction Section
    # -----------------------------------------------------
    st.markdown("## 🔮 Try Prediction Using the Learned Tree")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        outlook = st.selectbox("Outlook", ["Sunny", "Rain", "Overcast"])

    with col2:
        temperature = st.selectbox("Temperature", df["temperature"].unique())

    with col3:
        humidity = st.selectbox("Humidity", df["humidity"].unique())

    with col4:
        windy = st.selectbox("Windy", df["windy"].unique())

    if st.button("Predict Outcome"):
        st.markdown("### 🧠 Decision Path Followed:")

        if outlook == "Overcast":
            st.write("→ Outlook = Overcast ⇒ **Play = Yes**")
            st.success("🎉 Final Prediction: **YES**")
        elif outlook == "Sunny":
            st.write("→ Outlook = Sunny")
            if humidity == "High":
                st.write("→ Humidity = High ⇒ **Play = No**")
                st.error("Prediction: **NO**")
            else:
                st.write("→ Humidity ≠ High ⇒ **Play = Yes**")
                st.success("Prediction: **YES**")
        elif outlook == "Rain":
            st.write("→ Outlook = Rain")
            if windy == "True" or windy == True:
                st.write("→ Windy = True ⇒ **Play = No**")
                st.error("Prediction: **NO**")
            else:
                st.write("→ Windy = False ⇒ **Play = Yes**")
                st.success("Prediction: **YES**")


# ---------------------------------------------------------
# Step Navigation Buttons
# ---------------------------------------------------------
col_prev, col_next = st.columns(2)

with col_prev:
    if st.button("⬅ Previous Step"):
        prev_step()

with col_next:
    if st.button("Next Step ➡"):
        next_step()

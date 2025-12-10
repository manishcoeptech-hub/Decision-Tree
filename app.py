import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🌳 ID3 Decision Tree — Step-by-Step Simulator (Weather Dataset)")

# ==========================================================
# Utility Functions
# ==========================================================

def entropy(counts):
    """Compute entropy given class counts."""
    total = sum(counts.values())
    ent = 0
    for c in counts.values():
        if c != 0:
            p = c / total
            ent -= p * math.log2(p)
    return ent


def compute_information_gain(df, feature, target="play"):
    """Compute the IG of splitting df on 'feature'."""
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


def draw_tree(nodes, edges, title="Decision Tree"):
    """
    Draws a compact tree.
    nodes = dict(node_id -> {"text": "...", "x": float, "y": float})
    edges = list of (parent_id, child_id)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(title, fontsize=16)
    ax.axis("off")

    # Draw edges first
    for (p, c) in edges:
        x1, y1 = nodes[p]["x"], nodes[p]["y"]
        x2, y2 = nodes[c]["x"], nodes[c]["y"]
        ax.plot([x1, x2], [y1, y2], color="black")

    # Draw nodes
    for node_id, node in nodes.items():
        x, y = node["x"], node["y"]
        text = node["text"]

        ax.text(
            x, y, text,
            ha="center", va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#eef", edgecolor="black")
        )

    return fig


# ==========================================================
# Step Management
# ==========================================================
if "step" not in st.session_state:
    st.session_state.step = 1

def next_step():
    if st.session_state.step < 12:
        st.session_state.step += 1

def prev_step():
    if st.session_state.step > 1:
        st.session_state.step -= 1


# ==========================================================
# FILE UPLOAD
# ==========================================================
uploaded = st.file_uploader("Upload Weather CSV (Outlook,Temperature,Humidity,Windy,Play)", type=["csv"])

if not uploaded:
    st.warning("Please upload the CSV to begin.")
    st.stop()

df = pd.read_csv(uploaded)
df.columns = df.columns.str.strip().str.lower()

required = ["outlook","temperature","humidity","windy","play"]
if not all(col in df.columns for col in required):
    st.error("CSV must contain: Outlook, Temperature, Humidity, Windy, Play")
    st.stop()


# ==========================================================
# MAIN 12-STEP LOGIC
# ==========================================================
step = st.session_state.step

st.subheader(f"🪜 Step {step} of 12")

# ----------------------------
# STEP 1: Parent Entropy
# ----------------------------
if step == 1:
    st.markdown("### 📌 Step 1 — Compute Parent Entropy")
    counts = df["play"].value_counts().to_dict()

    st.write("#### Class Counts:")
    st.write(counts)

    ent = entropy(counts)
    st.latex(r"Entropy = -\sum p_i \log_2(p_i)")
    st.write(f"### 👉 Parent Entropy = **{ent:.4f}**")

    st.info("This entropy measures impurity of target variable (Play).")

# ----------------------------
# STEPS 2–5: Information Gain
# ----------------------------
features = ["outlook", "temperature", "humidity", "windy"]

if step in [2,3,4,5]:
    feature = features[step-2]
    st.markdown(f"### 📌 Step {step} — Compute Information Gain for **{feature.title()}**")

    parent_entropy, weighted_entropy, IG, subsets = compute_information_gain(df, feature)

    st.write("#### Subset Entropies:")
    for v, subset in subsets.items():
        c = subset["play"].value_counts().to_dict()
        st.write(f"**{feature.title()} = {v}** → Entropy = {entropy(c):.4f} (samples={len(subset)})")

    st.write("---")
    st.write(f"Weighted Entropy = **{weighted_entropy:.4f}**")
    st.write(f"### 👉 Information Gain = **{IG:.4f}**")

    st.info("Higher IG means the feature is better for splitting.")

# ----------------------------
# STEP 6: Choose Root Node
# ----------------------------
if step == 6:
    st.markdown("### 📌 Step 6 — Select Best Feature (Root)")

    IGs = {}
    for f in features:
        _, _, ig, _ = compute_information_gain(df, f)
        IGs[f] = ig

    st.write("### Information Gains:")
    st.write(IGs)

    root = max(IGs, key=IGs.get)
    st.success(f"🎉 Best Feature = **{root.title()}** (Root Node)")

    # Draw root-only tree
    nodes = {
        "root": {"text": f"{root.title()}\nEntropy={entropy(df['play'].value_counts()):.3f}",
                 "x": 0.5, "y": 0.9}
    }
    edges = []
    fig = draw_tree(nodes, edges, "Initial Root Node")
    st.pyplot(fig)

# ----------------------------
# STEP 7: Create Root Branches
# ----------------------------
if step == 7:
    st.markdown("### 📌 Step 7 — Create Root Branches")

    IGs = {f: compute_information_gain(df, f)[2] for f in features}
    root = max(IGs, key=IGs.get)

    children = df[root].unique()

    st.write(f"Root feature: **{root.title()}**")
    st.write("Branches:", children)

    # Draw partial tree
    nodes = {
        "root": {"text": root.title(), "x": 0.5, "y": 0.9}
    }
    edges = []

    x_positions = np.linspace(0.2, 0.8, len(children))
    for i, c in enumerate(children):
        nodes[c] = {"text": f"{c}", "x": x_positions[i], "y": 0.6}
        edges.append(("root", c))

    fig = draw_tree(nodes, edges, "Root With Branches")
    st.pyplot(fig)

# ----------------------------
# STEP 8–11: Sunny & Rain Subsets
# ----------------------------

# We reuse root
IGs = {f: compute_information_gain(df, f)[2] for f in features}
root = max(IGs, key=IGs.get)
subsets = {v: df[df[root] == v] for v in df[root].unique()}

# Sunny branch
if step == 8:
    st.markdown("### 📌 Step 8 — Compute Entropy of Sunny Subset")
    sunny = subsets.get("Sunny") or subsets.get("sunny")
    if sunny is None:
        st.error("Sunny not found in dataset.")
    else:
        c = sunny["play"].value_counts().to_dict()
        st.write("Sunny subset:", sunny)
        st.write("Counts:", c)
        st.write(f"### 👉 Entropy(Sunny) = **{entropy(c):.4f}**")

if step == 9:
    st.markdown("### 📌 Step 9 — Compute IG on Sunny Subset (Humidity Wins)")
    sunny = subsets.get("Sunny") or subsets.get("sunny")

    # Compute IG for Temperature/Humidity/Windy only
    st.write("Sunny subset IG calculations:")
    IG_sunny = {}
    for f in ["temperature","humidity","windy"]:
        _, _, ig, _ = compute_information_gain(sunny, f)
        IG_sunny[f] = ig
    st.write(IG_sunny)
    best = max(IG_sunny, key=IG_sunny.get)
    st.success(f"Best feature for Sunny = **{best.title()}**")

if step == 10:
    st.markdown("### 📌 Step 10 — Entropy of Rain Subset")
    rain = subsets.get("Rain") or subsets.get("rain")
    c = rain["play"].value_counts().to_dict()
    st.write(rain)
    st.write("Counts:", c)
    st.write(f"### 👉 Entropy(Rain) = **{entropy(c):.4f}**")

if step == 11:
    st.markdown("### 📌 Step 11 — Compute IG on Rain Subset (Windy Wins)")
    rain = subsets.get("Rain") or subsets.get("rain")
    IG_rain = {}
    for f in ["temperature","humidity","windy"]:
        _, _, ig, _ = compute_information_gain(rain, f)
        IG_rain[f] = ig
    st.write(IG_rain)
    best = max(IG_rain, key=IG_rain.get)
    st.success(f"Best feature for Rain = **{best.title()}**")

# ----------------------------
# STEP 12: Final Tree
# ----------------------------
if step == 12:
    st.markdown("### 🎉 Step 12 — Final ID3 Decision Tree")

    # Draw final compact tree
    nodes = {
        "root": {"text": "Outlook", "x": 0.5, "y": 0.9},
        "Sunny": {"text": "Sunny\n→ Humidity", "x": 0.25, "y": 0.6},
        "Rain": {"text": "Rain\n→ Windy", "x": 0.5, "y": 0.6},
        "Overcast": {"text": "Overcast\n→ Yes", "x": 0.75, "y": 0.6},
        "High": {"text": "High → No", "x": 0.15, "y": 0.3},
        "Normal": {"text": "Normal → Yes", "x": 0.35, "y": 0.3},
        "WindyTrue": {"text": "Windy=True → No", "x": 0.45, "y": 0.3},
        "WindyFalse": {"text": "Windy=False → Yes", "x": 0.55, "y": 0.3},
    }

    edges = [
        ("root","Sunny"), ("root","Rain"), ("root","Overcast"),
        ("Sunny","High"), ("Sunny","Normal"),
        ("Rain","WindyTrue"), ("Rain","WindyFalse")
    ]

    fig = draw_tree(nodes, edges, "Final ID3 Tree")
    st.pyplot(fig)

    st.success("Tree construction complete! Now students understand ID3 step-by-step.")

# ==========================================================
# NAVIGATION
# ==========================================================
col1, col2 = st.columns(2)
with col1:
    if st.button("⬅ Previous Step"):
        prev_step()
with col2:
    if st.button("Next Step ➡"):
        next_step()

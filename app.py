import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

st.set_page_config(page_title="ID3 Decision Tree Simulator",
                   layout="wide")

st.title("🌳 ID3 Decision Tree — Step-by-Step Simulation (Auto-Layout Tree)")

# ============================================================
# ENTROPY FUNCTION
# ============================================================
def entropy(counts):
    """
    counts = dict {"Yes": 9, "No": 5}
    Returns Shannon entropy.
    """
    total = sum(counts.values())
    if total == 0:
        return 0
    ent = 0
    for c in counts.values():
        if c > 0:
            p = c / total
            ent -= p * math.log2(p)
    return ent


# ============================================================
# INFORMATION GAIN CALCULATION
# ============================================================
def compute_information_gain(df, feature, target="play"):
    """
    Computes:
      - parent entropy
      - weighted entropy
      - information gain
      - subsets for each value
    """
    parent_counts = df[target].value_counts().to_dict()
    parent_entropy = entropy(parent_counts)

    subsets = {}
    weighted_entropy = 0

    for val in df[feature].unique():
        subset = df[df[feature] == val]
        subsets[val] = subset

        subset_counts = subset[target].value_counts().to_dict()
        weighted_entropy += (len(subset) / len(df)) * entropy(subset_counts)

    IG = parent_entropy - weighted_entropy
    return parent_entropy, weighted_entropy, IG, subsets


# ============================================================
# AUTO-LAYOUT TREE DRAWING ENGINE
# ============================================================
def compute_positions(tree, x=0.5, y=1.0, level_gap=0.18):
    """
    Recursively compute x/y positions for auto-spaced tree.

    tree = {
       "node_id": {
          "text": "...",
          "children": ["child1", "child2"]
       }
    }

    Returns dict positions[node_id] = (x,y)
    """
    positions = {}

    def dfs(node, depth, offset):
        children = tree[node].get("children", [])
        positions[node] = (offset, 1 - depth * level_gap)

        if len(children) == 0:
            return 1

        total_width = 0
        child_widths = []

        for child in children:
            w = dfs(child, depth + 1, offset)
            child_widths.append(w)
            offset += w
            total_width += w

        # Recenter children under parent
        if total_width > 0:
            parent_x = sum(positions[c][0] for c in children) / len(children)
            y = positions[node][1]
            positions[node] = (parent_x, y)

        return total_width

    dfs("root", 0, x)
    return positions


def draw_auto_tree(tree, title="Decision Tree"):
    """
    Draw tree with automatically spaced nodes.
    tree must contain:
        tree[node_id]["text"]
        tree[node_id]["children"]
    """
    positions = compute_positions(tree)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_title(title, fontsize=18)
    ax.axis("off")

    # Draw edges first
    for node, data in tree.items():
        for child in data.get("children", []):
            x1, y1 = positions[node]
            x2, y2 = positions[child]
            ax.plot([x1, x2], [y1, y2], color="black")

    # Draw nodes
    for node, (x, y) in positions.items():
        text = tree[node]["text"]
        ax.text(
            x, y, text,
            ha="center", va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="#eef",
                      edgecolor="black")
        )

    return fig

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
if "step" not in st.session_state:
    st.session_state.step = 1

if "ig_root" not in st.session_state:
    st.session_state.ig_root = {}

if "ig_sunny" not in st.session_state:
    st.session_state.ig_sunny = {}

if "ig_rain" not in st.session_state:
    st.session_state.ig_rain = {}

if "subsets" not in st.session_state:
    st.session_state.subsets = {}

if "root_feature" not in st.session_state:
    st.session_state.root_feature = None


# ============================================================
# NAVIGATION BUTTONS
# ============================================================
def next_step():
    if st.session_state.step < 12:
        st.session_state.step += 1

def prev_step():
    if st.session_state.step > 1:
        st.session_state.step -= 1


# ============================================================
# FILE UPLOAD SECTION
# ============================================================
uploaded = st.file_uploader(
    "📄 Upload Weather Dataset (Outlook, Temperature, Humidity, Windy, Play)",
    type=["csv"]
)

if not uploaded:
    st.warning("Please upload a CSV file to begin.")
    st.stop()

df = pd.read_csv(uploaded)
df.columns = df.columns.str.strip().str.lower()

required_cols = ["outlook", "temperature", "humidity", "windy", "play"]

if not all(col in df.columns for col in required_cols):
    st.error(f"CSV must contain these columns: {required_cols}")
    st.stop()


# ============================================================
# STEP ENGINE
# ============================================================
step = st.session_state.step
st.subheader(f"🪜 Step {step} of 12")


# ============================================================
# STEP 1 — Parent Entropy
# ============================================================
if step == 1:
    st.markdown("## 📌 Step 1 — Compute Parent Entropy")

    counts = df["play"].value_counts().to_dict()
    st.write("### Class Counts:", counts)

    parent_ent = entropy(counts)
    st.write(f"### 👉 Parent Entropy = **{parent_ent:.4f}**")

    st.info("Entropy measures dataset impurity. Next, we compute Information Gain for each feature.")


# ============================================================
# FEATURES FOR ROOT
# ============================================================
features = ["outlook", "temperature", "humidity", "windy"]


# ============================================================
# STEP 2 — IG for Outlook
# ============================================================
if step == 2:
    st.markdown("## 📌 Step 2 — Information Gain for **Outlook**")

    parent_entropy, weighted_entropy, IG, subsets = compute_information_gain(df, "outlook")
    st.session_state.ig_root["outlook"] = IG

    st.write("### Subset Entropies")

    for value, subset in subsets.items():
        c = subset["play"].value_counts().to_dict()
        st.write(f"- **{value}** → {c}, Entropy = {entropy(c):.4f}")

    st.write("---")
    st.write(f"Weighted Entropy = **{weighted_entropy:.4f}**")
    st.write(f"### 👉 IG(Outlook) = **{IG:.4f}**")


# ============================================================
# STEP 3 — IG for Temperature
# ============================================================
if step == 3:
    st.markdown("## 📌 Step 3 — Information Gain for **Temperature**")

    parent_entropy, weighted_entropy, IG, subsets = compute_information_gain(df, "temperature")
    st.session_state.ig_root["temperature"] = IG

    st.write("### Subset Entropies")

    for value, subset in subsets.items():
        c = subset["play"].value_counts().to_dict()
        st.write(f"- **{value}** → {c}, Entropy = {entropy(c):.4f}")

    st.write("---")
    st.write(f"Weighted Entropy = **{weighted_entropy:.4f}**")
    st.write(f"### 👉 IG(Temperature) = **{IG:.4f}**")


# ============================================================
# STEP 4 — IG for Humidity
# ============================================================
if step == 4:
    st.markdown("## 📌 Step 4 — Information Gain for **Humidity**")

    parent_entropy, weighted_entropy, IG, subsets = compute_information_gain(df, "humidity")
    st.session_state.ig_root["humidity"] = IG

    st.write("### Subset Entropies")

    for value, subset in subsets.items():
        c = subset["play"].value_counts().to_dict()
        st.write(f"- **{value}** → {c}, Entropy = {entropy(c):.4f}")

    st.write("---")
    st.write(f"Weighted Entropy = **{weighted_entropy:.4f}**")
    st.write(f"### 👉 IG(Humidity) = **{IG:.4f}**")


# ============================================================
# STEP 5 — IG for Windy
# ============================================================
if step == 5:
    st.markdown("## 📌 Step 5 — Information Gain for **Windy**")

    parent_entropy, weighted_entropy, IG, subsets = compute_information_gain(df, "windy")
    st.session_state.ig_root["windy"] = IG

    st.write("### Subset Entropies")

    for value, subset in subsets.items():
        c = subset["play"].value_counts().to_dict()
        st.write(f"- **{value}** → {c}, Entropy = {entropy(c):.4f}")

    st.write("---")
    st.write(f"Weighted Entropy = **{weighted_entropy:.4f}**")
    st.write(f"### 👉 IG(Windy) = **{IG:.4f}**")

# ============================================================
# STEP 6 — Select Best Feature as Root Node
# ============================================================
if step == 6:
    st.markdown("## 📌 Step 6 — Select Best Feature (Root Node)")

    IGs = st.session_state.ig_root

    st.write("### Information Gains Computed So Far:")
    st.write(IGs)

    # Best feature
    root_feature = max(IGs, key=IGs.get)
    st.session_state.root_feature = root_feature

    st.success(f"🎉 Best Feature = **{root_feature.title()}** (Root Node)")

    st.info("This feature has the maximum Information Gain, so it becomes the root of the tree.")


# ============================================================
# STEP 7 — Split Dataset into Branches Based on Root Feature
# ============================================================
if step == 7:
    root = st.session_state.root_feature
    st.markdown(f"## 📌 Step 7 — Split Data on Root Feature **{root.title()}**")

    # Unique values become branches
    branch_values = df[root].unique().tolist()
    st.write("### Branch Values:", branch_values)

    subsets = {}
    for val in branch_values:
        subset = df[df[root] == val]
        subsets[val] = subset
        st.write(f"### 👉 Branch '{val}' (Samples = {len(subset)})")
        st.dataframe(subset)

    st.session_state.subsets = subsets

    st.info("""
    Each unique value under the root feature creates a branch.
    Next, we analyze the **Sunny** and **Rain** subsets to build the tree layer by layer.
    """)

# ============================================================
# STEP 8 — Entropy of Sunny Subset
# ============================================================
if step == 8:
    st.markdown("## 📌 Step 8 — Compute Entropy of **Sunny** Subset")

    subsets = st.session_state.subsets

    # Locate Sunny subset (case-insensitive)
    sunny_key = next((k for k in subsets if k.lower() == "sunny"), None)
    sunny = subsets.get(sunny_key)

    if sunny is None:
        st.error("Sunny subset not found! Check if 'Outlook' contains 'Sunny' or similar values.")
    else:
        st.write("### 🌞 Sunny Subset")
        st.dataframe(sunny)

        counts = sunny["play"].value_counts().to_dict()
        st.write("### Class Counts:", counts)

        ent_sunny = entropy(counts)
        st.write(f"### 👉 Entropy(Sunny) = **{ent_sunny:.4f}**")

        if ent_sunny == 0:
            st.success("Sunny subset is PURE — no further splitting needed.")
        else:
            st.info("Sunny subset is **not pure**, so we compute IG for its remaining features next.")


# ============================================================
# STEP 9 — IG inside Sunny Subset
# ============================================================
if step == 9:
    st.markdown("## 📌 Step 9 — Compute Information Gain for **Sunny** Branch")

    subsets = st.session_state.subsets
    sunny_key = next((k for k in subsets if k.lower() == "sunny"), None)
    sunny = subsets.get(sunny_key)

    st.write("### 🌞 Sunny Subset Data")
    st.dataframe(sunny)

    st.markdown("### Remaining Features to Evaluate:")
    st.write("- Temperature")
    st.write("- Humidity")
    st.write("- Windy")

    features_remaining = ["temperature", "humidity", "windy"]
    IG_sunny = {}

    for f in features_remaining:
        _, _, ig, _ = compute_information_gain(sunny, f)
        IG_sunny[f] = ig

    st.session_state.ig_sunny = IG_sunny

    st.write("### Information Gains in Sunny Subset:")
    st.write(IG_sunny)

    best = max(IG_sunny, key=IG_sunny.get)
    st.success(f"🎉 Best Feature for Sunny = **{best.title()}**")


# ============================================================
# STEP 10 — Entropy of Rain Subset
# ============================================================
if step == 10:
    st.markdown("## 📌 Step 10 — Compute Entropy of **Rain** Subset")

    subsets = st.session_state.subsets
    rain_key = next((k for k in subsets if k.lower() == "rain"), None)
    rain = subsets.get(rain_key)

    st.write("### 🌧 Rain Subset")
    st.dataframe(rain)

    counts = rain["play"].value_counts().to_dict()
    st.write("### Class Counts:", counts)

    ent_rain = entropy(counts)
    st.write(f"### 👉 Entropy(Rain) = **{ent_rain:.4f}**")

    if ent_rain == 0:
        st.success("Rain subset is PURE — no further splitting needed.")
    else:
        st.info("Rain subset is **not pure**, so we compute IG next.")


# ============================================================
# STEP 11 — IG inside Rain Subset
# ============================================================
if step == 11:
    st.markdown("## 📌 Step 11 — Compute Information Gain for **Rain** Branch")

    subsets = st.session_state.subsets
    rain_key = next((k for k in subsets if k.lower() == "rain"), None)
    rain = subsets.get(rain_key)

    st.write("### 🌧 Rain Subset Data")
    st.dataframe(rain)

    features_remaining = ["temperature", "humidity", "windy"]
    IG_rain = {}

    for f in features_remaining:
        _, _, ig, _ = compute_information_gain(rain, f)
        IG_rain[f] = ig

    st.session_state.ig_rain = IG_rain

    st.write("### Information Gains in Rain Subset:")
    st.write(IG_rain)

    best = max(IG_rain, key=IG_rain.get)
    st.success(f"🎉 Best Feature for Rain = **{best.title()}**")

    st.info("We now have enough information to build the **final tree**.")

# ============================================================
# STEP 12 — Final Auto-Layout Decision Tree
# ============================================================
if step == 12:
    st.markdown("## 🎉 Step 12 — Final ID3 Decision Tree (Auto-Layout)")
    st.markdown("""
    The tree is now constructed using the best splits found earlier.
    We use an **auto-layout algorithm** to space nodes cleanly.
    """)

    # --------------------------------------------------------
    # Build full tree structure
    # --------------------------------------------------------
    tree = {
        "root": {
            "text": "Outlook",
            "children": ["Sunny", "Rain", "Overcast"]
        },

        # -- Sunny branch --
        "Sunny": {
            "text": f"Sunny → {max(st.session_state.ig_sunny, key=st.session_state.ig_sunny.get).title()}",
            "children": ["High", "Normal"]
        },
        "High": {
            "text": "High → No",
            "children": []
        },
        "Normal": {
            "text": "Normal → Yes",
            "children": []
        },

        # -- Rain branch --
        "Rain": {
            "text": f"Rain → {max(st.session_state.ig_rain, key=st.session_state.ig_rain.get).title()}",
            "children": ["WindyTrue", "WindyFalse"]
        },
        "WindyTrue": {
            "text": "Windy=True → No",
            "children": []
        },
        "WindyFalse": {
            "text": "Windy=False → Yes",
            "children": []
        },

        # -- Overcast branch --
        "Overcast": {
            "text": "Overcast → Yes",
            "children": []
        }
    }

    # --------------------------------------------------------
    # Draw Tree
    # --------------------------------------------------------
    fig = draw_auto_tree(tree, "Final ID3 Decision Tree")
    st.pyplot(fig)

    st.success("🎯 Tree Construction Complete!")


    # ========================================================
    # PREDICTION UI
    # ========================================================
    st.markdown("## 🔮 Try Prediction Using This Tree")

    col1, col2, col3, col4 = st.columns(4)
    outlook = col1.selectbox("Outlook", df["outlook"].unique())
    temperature = col2.selectbox("Temperature", df["temperature"].unique())
    humidity = col3.selectbox("Humidity", df["humidity"].unique())
    windy = col4.selectbox("Windy", df["windy"].unique())

    if st.button("Predict Outcome"):
        st.markdown("### 🧠 Decision Path:")

        # --- Overcast branch ---
        if outlook.lower() == "overcast":
            st.write("→ Outlook = Overcast ⇒ **Play = Yes**")
            st.success("Prediction: YES")

        # --- Sunny branch ---
        elif outlook.lower() == "sunny":
            st.write("→ Outlook = Sunny")
            best_sunny = max(st.session_state.ig_sunny, key=st.session_state.ig_sunny.get)

            if best_sunny == "humidity":
                st.write("→ Split on Humidity")

                if str(humidity).lower() == "high":
                    st.write("→ Humidity = High ⇒ **Play = No**")
                    st.error("Prediction: NO")
                else:
                    st.write("→ Humidity = Normal ⇒ **Play = Yes**")
                    st.success("Prediction: YES")

        # --- Rain branch ---
        elif outlook.lower() == "rain":
            st.write("→ Outlook = Rain")
            best_rain = max(st.session_state.ig_rain, key=st.session_state.ig_rain.get)

            if best_rain == "windy":
                st.write("→ Split on Windy")

                if str(windy).lower() in ["true", "yes", "1"]:
                    st.write("→ Windy = True ⇒ **Play = No**")
                    st.error("Prediction: NO")
                else:
                    st.write("→ Windy = False ⇒ **Play = Yes**")
                    st.success("Prediction: YES")


# ============================================================
# NAVIGATION CONTROLS (always visible)
# ============================================================
col_prev, col_next = st.columns(2)

with col_prev:
    if st.button("⬅ Previous Step"):
        prev_step()

with col_next:
    if st.button("Next Step ➡"):
        next_step()

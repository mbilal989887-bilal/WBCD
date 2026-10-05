import pandas as pd
import matplotlib.pyplot as plt
import io, base64

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

# ================================
# LOAD & TRAIN MODEL
# ================================
df = pd.read_csv("data.csv")
df.columns = df.columns.str.strip().str.lower()
df.drop("id", axis=1, inplace=True)

df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    {
        "max_depth": [3, 5, 7],
        "criterion": ["gini", "entropy"]
    },
    cv=5
)

grid.fit(X_train, y_train)
model = grid.best_estimator_

# ================================
# UTIL FUNCTION
# ================================
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)
    return img

# ================================
# PREDICTION
# ================================
def predict_from_input(form_data):
    # Fill missing features with dataset means
    full_data = X.mean().to_dict()
    full_data.update(form_data)
    input_df = pd.DataFrame([full_data], columns=X.columns)
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]
    return pred, prob, input_df

# ================================
# GRAPH 1: PROBABILITY
# ================================
def probability_plot(prob):
    fig, ax = plt.subplots()
    ax.bar(["Benign", "Malignant"], [1 - prob, prob])
    ax.set_ylim(0, 1)
    ax.set_title("Prediction Probability")
    return fig_to_base64(fig)

# ================================
# GRAPH 2: USER VS MEAN
# ================================
def user_vs_mean_plot(input_df):
    feature = X.columns[0]
    fig, ax = plt.subplots()
    ax.bar(
        ["Dataset Mean", "User Value"],
        [X[feature].mean(), input_df[feature].values[0]]
    )
    ax.set_title(f"{feature} Comparison")
    return fig_to_base64(fig)

# ================================
# GRAPH 3: FEATURE IMPORTANCE
# ================================
def feature_importance_plot():
    importances = pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False).head(5)

    fig, ax = plt.subplots()
    importances.plot(kind="barh", ax=ax)
    ax.set_title("Top 5 Important Features")
    ax.invert_yaxis()
    return fig_to_base64(fig)

# ================================
# GET TOP FEATURES
# ================================
def get_top_features(n=5):
    importances = pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)
    return importances.head(n).index.tolist()

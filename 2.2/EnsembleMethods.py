"""
Attention: This is the alpha version, and it may contain several errors.
    Not all parts of the program tested carefully.

PROGRAM: Ensemble Methods Demonstration (Classification, Regression, Clustering, Dimensionality Reduction, Anomaly Detection)
AUTHOR:  Dr. Aliasghar Khavasi
DATE:    2026 - February

DESCRIPTION:
  A Tkinter-based GUI that introduces and demonstrates Ensemble Methods across multiple tasks:
    1) Classification
    2) Regression
    3) Clustering
    4) Dimensionality Reduction
    5) Anomaly Detection

  Users can:
    - Select built-in datasets or load their own CSV.
    - Adjust hyperparameters for each ensemble or classical model.
    - Fit the chosen model and visualize results in 2D (when feasible).
    - Test a single sample for supervised tasks.
    - Read two text memos:
        * Memo #1: Detailed learning objectives, references, and background.
        * Memo #2: Logs each step (data loading, hyperparameters, fitting, single sample testing, etc.).

  (Note: The Wine dataset has been removed in favor of datasets more suitable for demonstrating ensemble methods.)
"""

import tkinter as tk
from tkinter import ttk, simpledialog, messagebox, filedialog
import numpy as np
import pandas as pd
import threading
import time
import os
import tempfile

# Import datasets and models from scikit-learn and xgboost (if installed)
from sklearn.datasets import load_breast_cancer, load_iris, load_digits, fetch_california_housing, load_diabetes, make_blobs, make_moons
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor, IsolationForest
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    XGBClassifier, XGBRegressor = None, None

# GLOBAL PARAMETERS
DEFAULT_ROWS = 10
DEFAULT_COLS = 10

# For reproducibility
np.random.seed(42)

# --- Data dictionary ---
DATASETS = {
    "Classification": [
        ("Breast Cancer", lambda: load_breast_cancer(return_X_y=True)),
        ("Iris", lambda: load_iris(return_X_y=True)),
        ("Digits", lambda: load_digits(return_X_y=True))
    ],
    "Regression": [
        ("California Housing", lambda: (fetch_california_housing().data, fetch_california_housing().target)),
        ("Diabetes", lambda: load_diabetes(return_X_y=True))
    ],
    "Clustering": [
        ("Blobs", lambda: make_blobs(n_samples=300, centers=4, random_state=42)[0]),
        ("Moons", lambda: make_moons(n_samples=300, noise=0.1, random_state=42)[0]),
        ("Iris", lambda: load_iris(return_X_y=True)[0])
    ],
    "Dimensionality Reduction": [
        ("Digits", lambda: load_digits(return_X_y=True)[0]),
        ("Blobs", lambda: make_blobs(n_samples=300, centers=4, random_state=42)[0]),
        ("Moons", lambda: make_moons(n_samples=300, noise=0.1, random_state=42)[0])
    ],
    "Anomaly Detection": [
        ("Breast Cancer", lambda: load_breast_cancer(return_X_y=True)[0]),
        ("Blobs (with outliers)", lambda: (
            np.vstack([make_blobs(n_samples=300, centers=3, random_state=42)[0],
                       np.random.uniform(low=-10, high=10, size=(10, 2))])
        ))
    ]
}

# --- Ensemble algorithms available per task ---
ALGO_OPTIONS = {
    "Classification": ["Random Forest", "XGBoost", "AdaBoost"],
    "Regression": ["Random Forest Regressor", "XGBoost Regressor", "AdaBoost Regressor"],
    "Clustering": ["K-Means", "DBSCAN", "Agglomerative"],
    "Dimensionality Reduction": ["PCA", "t-SNE"],
    "Anomaly Detection": ["Isolation Forest", "Local Outlier Factor"]
}

# --- Default hyperparameter values per algorithm ---
DEFAULT_PARAMS = {
    "Random Forest": {"n_estimators": 100, "max_depth": 5, "min_samples_split": 2},
    "XGBoost": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 5} if XGBClassifier else {},
    "AdaBoost": {"n_estimators": 50, "learning_rate": 1.0},
    "Random Forest Regressor": {"n_estimators": 100, "max_depth": 5, "min_samples_split": 2},
    "XGBoost Regressor": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 5} if XGBRegressor else {},
    "AdaBoost Regressor": {"n_estimators": 50, "learning_rate": 1.0},
    "K-Means": {"n_clusters": 4, "init": "k-means++", "max_iter": 300, "n_init": 10},
    "DBSCAN": {"eps": 0.5, "min_samples": 5, "metric": "euclidean", "leaf_size": 30},
    "Agglomerative": {"n_clusters": 4, "linkage": "ward", "metric": "euclidean"},
    "PCA": {"n_components": 2, "whiten": False},
    "t-SNE": {"perplexity": 30.0, "n_iter": 1000, "learning_rate": 200.0},
    "Isolation Forest": {"n_estimators": 100, "contamination": 0.1},
    "Local Outlier Factor": {"n_neighbors": 20, "contamination": 0.1}
}

# --- Simple Ensemble Demo Core ---
class EnsembleDemoCore:
    def __init__(self, task, algo, hyperparams, X, y=None):
        self.task = task
        self.algo = algo
        self.hyperparams = hyperparams
        self.X = X
        self.y = y
        self.model = None
        self.embedding_ = None  # for t-SNE
        self.fit_model()

    def fit_model(self):
        if self.task in ["Classification", "Regression"]:
            if self.algo == "Random Forest":
                if self.task == "Classification":
                    from sklearn.ensemble import RandomForestClassifier
                    self.model = RandomForestClassifier(
                        n_estimators=self.hyperparams.get("n_estimators", 100),
                        max_depth=self.hyperparams.get("max_depth", 5),
                        min_samples_split=self.hyperparams.get("min_samples_split", 2),
                        random_state=42
                    )
                else:
                    from sklearn.ensemble import RandomForestRegressor
                    self.model = RandomForestRegressor(
                        n_estimators=self.hyperparams.get("n_estimators", 100),
                        max_depth=self.hyperparams.get("max_depth", 5),
                        min_samples_split=self.hyperparams.get("min_samples_split", 2),
                        random_state=42
                    )
            elif self.algo == "XGBoost":
                if XGBClassifier is None:
                    raise ImportError("xgboost is not installed.")
                if self.task == "Classification":
                    self.model = XGBClassifier(
                        n_estimators=self.hyperparams.get("n_estimators", 100),
                        learning_rate=self.hyperparams.get("learning_rate", 0.1),
                        max_depth=self.hyperparams.get("max_depth", 5),
                        random_state=42,
                        use_label_encoder=False,
                        eval_metric='logloss'
                    )
                else:
                    self.model = XGBRegressor(
                        n_estimators=self.hyperparams.get("n_estimators", 100),
                        learning_rate=self.hyperparams.get("learning_rate", 0.1),
                        max_depth=self.hyperparams.get("max_depth", 5),
                        random_state=42
                    )
            elif self.algo == "AdaBoost":
                if self.task == "Classification":
                    from sklearn.ensemble import AdaBoostClassifier
                    self.model = AdaBoostClassifier(
                        n_estimators=self.hyperparams.get("n_estimators", 50),
                        learning_rate=self.hyperparams.get("learning_rate", 1.0),
                        random_state=42
                    )
                else:
                    from sklearn.ensemble import AdaBoostRegressor
                    self.model = AdaBoostRegressor(
                        n_estimators=self.hyperparams.get("n_estimators", 50),
                        learning_rate=self.hyperparams.get("learning_rate", 1.0),
                        random_state=42
                    )
            if self.y is not None:
                self.model.fit(self.X, self.y)
            else:
                self.model.fit(self.X)
        elif self.task == "Clustering":
            if self.algo == "K-Means":
                from sklearn.cluster import KMeans
                self.model = KMeans(
                    n_clusters=self.hyperparams.get("n_clusters", 4),
                    init=self.hyperparams.get("init", "k-means++"),
                    max_iter=self.hyperparams.get("max_iter", 300),
                    n_init=self.hyperparams.get("n_init", 10),
                    random_state=42
                )
                self.model.fit(self.X)
            elif self.algo == "DBSCAN":
                from sklearn.cluster import DBSCAN
                self.model = DBSCAN(
                    eps=self.hyperparams.get("eps", 0.5),
                    min_samples=self.hyperparams.get("min_samples", 5),
                    metric=self.hyperparams.get("metric", "euclidean"),
                    leaf_size=self.hyperparams.get("leaf_size", 30)
                )
                self.model.fit(self.X)
            elif self.algo == "Agglomerative":
                from sklearn.cluster import AgglomerativeClustering
                n_clusters = self.hyperparams.get("n_clusters", 4)
                linkage = self.hyperparams.get("linkage", "ward")
                if linkage == "ward":
                    self.model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
                else:
                    metric = self.hyperparams.get("metric", "euclidean")
                    self.model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, metric=metric)
                self.model.fit(self.X)
        elif self.task == "Dimensionality Reduction":
            if self.algo == "PCA":
                self.model = PCA(
                    n_components=self.hyperparams.get("n_components", 2),
                    whiten=self.hyperparams.get("whiten", False),
                    random_state=42
                )
                self.model.fit(self.X)
            elif self.algo == "t-SNE":
                self.model = TSNE(
                    n_components=2,
                    perplexity=self.hyperparams.get("perplexity", 30.0),
                    max_iter=self.hyperparams.get("n_iter", 1000),
                    learning_rate=self.hyperparams.get("learning_rate", 200.0),
                    random_state=42
                )
                self.embedding_ = self.model.fit_transform(self.X)
        elif self.task == "Anomaly Detection":
            if self.algo == "Isolation Forest":
                self.model = IsolationForest(
                    n_estimators=self.hyperparams.get("n_estimators", 100),
                    contamination=self.hyperparams.get("contamination", 0.1),
                    random_state=42
                )
                self.model.fit(self.X)
            elif self.algo == "Local Outlier Factor":
                self.model = LocalOutlierFactor(
                    n_neighbors=self.hyperparams.get("n_neighbors", 20),
                    contamination=self.hyperparams.get("contamination", 0.1)
                )
                self.lof_labels = self.model.fit_predict(self.X)

    def predict_sample(self, sample):
        if self.task in ["Classification", "Regression"]:
            return self.model.predict(sample)
        elif self.task == "Clustering":
            if hasattr(self.model, "predict"):
                return self.model.predict(sample)
            else:
                return None
        elif self.task == "Dimensionality Reduction":
            if self.algo == "PCA":
                return self.model.transform(sample)
            elif self.algo == "t-SNE":
                return None
        elif self.task == "Anomaly Detection":
            if self.algo == "Isolation Forest":
                return self.model.predict(sample)
            elif self.algo == "Local Outlier Factor":
                return None
        return None

# --- GUI for Ensemble Methods Demo ---
class EnsembleDemoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ensemble Methods Demo")
        self.root.geometry("1400x900")

        # Task type radio buttons
        self.task_var = tk.StringVar(value="Classification")
        task_frame = tk.Frame(root, bd=2, relief=tk.RIDGE)
        task_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        tk.Label(task_frame, text="Task Type:").pack(side=tk.LEFT, padx=5)
        for task in ["Classification", "Regression", "Clustering", "Dimensionality Reduction", "Anomaly Detection"]:
            tk.Radiobutton(task_frame, text=task, variable=self.task_var, value=task,
                           command=self.update_task_mode).pack(side=tk.LEFT, padx=5)

        # Dataset selection
        data_frame = tk.Frame(root, bd=2, relief=tk.RIDGE)
        data_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        tk.Label(data_frame, text="Dataset:").pack(side=tk.LEFT, padx=5)
        self.dataset_var = tk.StringVar(value="---")
        self.dataset_combo = ttk.Combobox(data_frame, textvariable=self.dataset_var, state="readonly", width=40)
        self.dataset_combo.pack(side=tk.LEFT, padx=5)
        tk.Button(data_frame, text="Load Built-in", command=self.load_builtin_dataset).pack(side=tk.LEFT, padx=5)
        tk.Button(data_frame, text="Load My CSV", command=self.load_csv).pack(side=tk.LEFT, padx=5)
        tk.Button(data_frame, text="Data Info", command=self.show_data_info).pack(side=tk.LEFT, padx=5)
        tk.Button(data_frame, text="Open in Excel", command=self.open_in_excel).pack(side=tk.LEFT, padx=5)

        # Algorithm and hyperparameters
        algo_frame = tk.Frame(root, bd=2, relief=tk.RIDGE)
        algo_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        tk.Label(algo_frame, text="Algorithm:").pack(side=tk.LEFT, padx=5)
        self.algo_var = tk.StringVar(value="---")
        self.algo_combo = ttk.Combobox(algo_frame, textvariable=self.algo_var, state="readonly", width=40)
        self.algo_combo.pack(side=tk.LEFT, padx=5)
        self.algo_combo.bind("<<ComboboxSelected>>", lambda e: self.build_hyperparams())
        self.hyperparams_frame = tk.Frame(algo_frame)
        self.hyperparams_frame.pack(side=tk.LEFT, padx=5)
        tk.Button(algo_frame, text="Fit Model", command=self.fit_model).pack(side=tk.LEFT, padx=5)
        tk.Button(algo_frame, text="Test Sample", command=self.test_sample).pack(side=tk.LEFT, padx=5)
        tk.Button(algo_frame, text="Visualize", command=self.visualize_model).pack(side=tk.LEFT, padx=5)

        # Training controls
        train_frame = tk.Frame(root, bd=2, relief=tk.RIDGE)
        train_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        tk.Label(train_frame, text="Speed (sec/iter):").pack(side=tk.LEFT)
        self.speed_var = tk.DoubleVar(value=0.5)
        tk.Scale(train_frame, variable=self.speed_var, from_=0.0, to=2.0, resolution=0.1,
                 orient=tk.HORIZONTAL, length=150).pack(side=tk.LEFT, padx=5)
        tk.Button(train_frame, text="Start", command=self.on_start).pack(side=tk.LEFT, padx=5)
        tk.Button(train_frame, text="Pause", command=self.on_pause).pack(side=tk.LEFT, padx=5)
        tk.Button(train_frame, text="Resume", command=self.on_resume).pack(side=tk.LEFT, padx=5)
        tk.Button(train_frame, text="Step", command=self.on_step).pack(side=tk.LEFT, padx=5)
        tk.Button(train_frame, text="Stop", command=self.on_stop).pack(side=tk.LEFT, padx=5)

        # Memos
        memo_frame = tk.Frame(root)
        memo_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.memo1_frame = tk.LabelFrame(memo_frame, text="Memo #1 (Learning Objectives)", padx=5, pady=5)
        self.memo1_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.memo1_text = tk.Text(self.memo1_frame, wrap=tk.WORD, height=15)
        self.memo1_text.pack(fill=tk.BOTH, expand=True)
        self.populate_memo1()  # Added method to fix error
        self.memo2_frame = tk.LabelFrame(memo_frame, text="Memo #2 (Behind the Scenes)", padx=5, pady=5)
        self.memo2_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.memo2_text = tk.Text(self.memo2_frame, wrap=tk.WORD, height=15)
        self.memo2_text.pack(fill=tk.BOTH, expand=True)

        # Model Output textual display
        self.qtable_frame = tk.LabelFrame(root, text="Model Output (Textual)", padx=5, pady=5)
        self.qtable_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.qtable_text = tk.Text(self.qtable_frame, wrap=tk.NONE, width=50, height=30)
        self.qtable_text.pack(fill=tk.BOTH, expand=True)

        # Variables to store dataset and ensemble core object
        self.df = None
        self.X = None
        self.y = None
        self.scaler = None
        self.model_core = None

        # Training thread
        self.training_thread = None
        self.running = False
        self.paused = False
        self.step_mode = False

        self.update_task_mode()
        self.log_memo2("Ensemble Methods Demo program initialized.\n")

    # ----------------------- Populate Memo #1 -----------------------
    def populate_memo1(self):
        text = (
            "ENSEMBLE METHODS DEMONSTRATION\n\n"
            "LEARNING OBJECTIVES:\n"
            "1) Explore ensemble methods for Classification, Regression, Clustering,\n"
            "   Dimensionality Reduction, and Anomaly Detection.\n"
            "2) Learn how to adjust hyperparameters and see their effects on model performance.\n"
            "3) Compare classical models (Random Forest, AdaBoost, XGBoost) and unsupervised methods.\n"
            "4) Visualize the results and test individual samples.\n\n"
            "FURTHER READING:\n"
            "• Scikit-Learn Ensembles: https://scikit-learn.org/stable/modules/ensemble.html\n"
            "• XGBoost Documentation: https://xgboost.readthedocs.io/en/latest/\n"
            "• Bagging vs Boosting: https://towardsdatascience.com/bagging-vs-boosting-9c91cd29c79f\n"
        )
        self.memo1_text.insert(tk.END, text)

    def clear_memo2(self):
        self.memo2_text.delete("1.0", tk.END)

    def log_memo2(self, msg):
        self.memo2_text.insert(tk.END, msg)
        self.memo2_text.see(tk.END)

    # ----------------------- Task and Dataset Updates -----------------------
    def update_task_mode(self):
        self.clear_memo2()
        task = self.task_var.get()
        self.log_memo2(f"Selected Task: {task}\n")
        datasets = DATASETS.get(task, [])
        ds_names = [name for name, _ in datasets]
        self.dataset_combo.config(values=ds_names)
        if ds_names:
            self.dataset_combo.set(ds_names[0])
        else:
            self.dataset_combo.set("---")
        algos = ALGO_OPTIONS.get(task, [])
        self.algo_combo.config(values=algos)
        if algos:
            self.algo_combo.set(algos[0])
        else:
            self.algo_combo.set("---")
        self.build_hyperparams()

    # ----------------------- Hyperparameter Panel -----------------------
    def build_hyperparams(self):
        for child in self.hyperparams_frame.winfo_children():
            child.destroy()
        task = self.task_var.get()
        algo = self.algo_var.get()
        if task in ["Classification", "Regression"]:
            if algo in ["Random Forest", "Random Forest Regressor"]:
                self._build_rf_ui()
            elif algo in ["XGBoost", "XGBoost Regressor"]:
                self._build_xgb_ui()
            elif algo in ["AdaBoost", "AdaBoost Regressor"]:
                self._build_adaboost_ui()
        elif task == "Clustering":
            if algo == "K-Means":
                self._build_kmeans_ui()
            elif algo == "DBSCAN":
                self._build_dbscan_ui()
            elif algo == "Agglomerative":
                self._build_agglo_ui()
        elif task == "Dimensionality Reduction":
            if algo == "PCA":
                self._build_pca_ui()
            elif algo == "t-SNE":
                self._build_tsne_ui()
        elif task == "Anomaly Detection":
            if algo == "Isolation Forest":
                self._build_isolationforest_ui()
            elif algo == "Local Outlier Factor":
                self._build_lof_ui()

    # ---------- Hyperparameter UI Builders for Supervised ----------
    def _build_rf_ui(self):
        row = 0
        tk.Label(self.hyperparams_frame, text="n_estimators:").grid(row=row, column=0, padx=2)
        self.rf_n_estimators_var = tk.IntVar(value=DEFAULT_PARAMS["Random Forest"].get("n_estimators", 100))
        tk.Entry(self.hyperparams_frame, textvariable=self.rf_n_estimators_var, width=5).grid(row=row, column=1, padx=2)
        row += 1
        tk.Label(self.hyperparams_frame, text="max_depth:").grid(row=row, column=0, padx=2)
        self.rf_max_depth_var = tk.IntVar(value=DEFAULT_PARAMS["Random Forest"].get("max_depth", 5))
        tk.Entry(self.hyperparams_frame, textvariable=self.rf_max_depth_var, width=5).grid(row=row, column=1, padx=2)
        row += 1
        tk.Label(self.hyperparams_frame, text="min_samples_split:").grid(row=row, column=0, padx=2)
        self.rf_min_split_var = tk.IntVar(value=DEFAULT_PARAMS["Random Forest"].get("min_samples_split", 2))
        tk.Entry(self.hyperparams_frame, textvariable=self.rf_min_split_var, width=5).grid(row=row, column=1, padx=2)

    def _build_xgb_ui(self):
        row = 0
        tk.Label(self.hyperparams_frame, text="n_estimators:").grid(row=row, column=0, padx=2)
        self.xgb_n_estimators_var = tk.IntVar(value=DEFAULT_PARAMS["XGBoost"].get("n_estimators", 100))
        tk.Entry(self.hyperparams_frame, textvariable=self.xgb_n_estimators_var, width=5).grid(row=row, column=1, padx=2)
        row += 1
        tk.Label(self.hyperparams_frame, text="learning_rate:").grid(row=row, column=0, padx=2)
        self.xgb_lr_var = tk.DoubleVar(value=DEFAULT_PARAMS["XGBoost"].get("learning_rate", 0.1))
        tk.Entry(self.hyperparams_frame, textvariable=self.xgb_lr_var, width=5).grid(row=row, column=1, padx=2)
        row += 1
        tk.Label(self.hyperparams_frame, text="max_depth:").grid(row=row, column=0, padx=2)
        self.xgb_max_depth_var = tk.IntVar(value=DEFAULT_PARAMS["XGBoost"].get("max_depth", 5))
        tk.Entry(self.hyperparams_frame, textvariable=self.xgb_max_depth_var, width=5).grid(row=row, column=1, padx=2)

    def _build_adaboost_ui(self):
        row = 0
        tk.Label(self.hyperparams_frame, text="n_estimators:").grid(row=row, column=0, padx=2)
        self.ab_n_estimators_var = tk.IntVar(value=DEFAULT_PARAMS["AdaBoost"].get("n_estimators", 50))
        tk.Entry(self.hyperparams_frame, textvariable=self.ab_n_estimators_var, width=5).grid(row=row, column=1, padx=2)
        row += 1
        tk.Label(self.hyperparams_frame, text="learning_rate:").grid(row=row, column=0, padx=2)
        self.ab_lr_var = tk.DoubleVar(value=DEFAULT_PARAMS["AdaBoost"].get("learning_rate", 1.0))
        tk.Entry(self.hyperparams_frame, textvariable=self.ab_lr_var, width=5).grid(row=row, column=1, padx=2)

    # ---------- Hyperparameter UI Builders for Clustering ----------
    def _build_kmeans_ui(self):
        row = 0
        tk.Label(self.hyperparams_frame, text="n_clusters:").grid(row=row, column=0, padx=2)
        self.km_n_clusters_var = tk.IntVar(value=DEFAULT_PARAMS["K-Means"].get("n_clusters", 4))
        tk.Entry(self.hyperparams_frame, textvariable=self.km_n_clusters_var, width=5).grid(row=row, column=1, padx=2)
        row += 1
        tk.Label(self.hyperparams_frame, text="init:").grid(row=row, column=0, padx=2)
        self.km_init_var = tk.StringVar(value=DEFAULT_PARAMS["K-Means"].get("init", "k-means++"))
        init_combo = ttk.Combobox(self.hyperparams_frame, textvariable=self.km_init_var, width=10,
                                  values=["k-means++", "random"])
        init_combo.grid(row=row, column=1, padx=2)
        row += 1
        tk.Label(self.hyperparams_frame, text="max_iter:").grid(row=row, column=0, padx=2)
        self.km_max_iter_var = tk.IntVar(value=DEFAULT_PARAMS["K-Means"].get("max_iter", 300))
        tk.Entry(self.hyperparams_frame, textvariable=self.km_max_iter_var, width=5).grid(row=row, column=1, padx=2)
        row += 1
        tk.Label(self.hyperparams_frame, text="n_init:").grid(row=row, column=0, padx=2)
        self.km_n_init_var = tk.IntVar(value=DEFAULT_PARAMS["K-Means"].get("n_init", 10))
        tk.Entry(self.hyperparams_frame, textvariable=self.km_n_init_var, width=5).grid(row=row, column=1, padx=2)

    def _build_dbscan_ui(self):
        row = 0
        tk.Label(self.hyperparams_frame, text="eps:").grid(row=row, column=0, padx=2)
        self.dbscan_eps_var = tk.DoubleVar(value=DEFAULT_PARAMS["DBSCAN"].get("eps", 0.5))
        tk.Entry(self.hyperparams_frame, textvariable=self.dbscan_eps_var, width=5).grid(row=row, column=1, padx=2)
        row += 1
        tk.Label(self.hyperparams_frame, text="min_samples:").grid(row=row, column=0, padx=2)
        self.dbscan_min_var = tk.IntVar(value=DEFAULT_PARAMS["DBSCAN"].get("min_samples", 5))
        tk.Entry(self.hyperparams_frame, textvariable=self.dbscan_min_var, width=5).grid(row=row, column=1, padx=2)
        row += 1
        tk.Label(self.hyperparams_frame, text="metric:").grid(row=row, column=0, padx=2)
        self.dbscan_metric_var = tk.StringVar(value=DEFAULT_PARAMS["DBSCAN"].get("metric", "euclidean"))
        met_combo = ttk.Combobox(self.hyperparams_frame, textvariable=self.dbscan_metric_var, width=10,
                                  values=["euclidean", "manhattan", "chebyshev", "cosine"])
        met_combo.grid(row=row, column=1, padx=2)
        row += 1
        tk.Label(self.hyperparams_frame, text="leaf_size:").grid(row=row, column=0, padx=2)
        self.dbscan_leaf_var = tk.IntVar(value=DEFAULT_PARAMS["DBSCAN"].get("leaf_size", 30))
        tk.Entry(self.hyperparams_frame, textvariable=self.dbscan_leaf_var, width=5).grid(row=row, column=1, padx=2)

    def _build_agglo_ui(self):
        row = 0
        tk.Label(self.hyperparams_frame, text="n_clusters:").grid(row=row, column=0, padx=2)
        self.agg_n_clusters_var = tk.IntVar(value=DEFAULT_PARAMS["Agglomerative"].get("n_clusters", 4))
        tk.Entry(self.hyperparams_frame, textvariable=self.agg_n_clusters_var, width=5).grid(row=row, column=1, padx=2)
        row += 1
        tk.Label(self.hyperparams_frame, text="linkage:").grid(row=row, column=0, padx=2)
        self.agg_linkage_var = tk.StringVar(value=DEFAULT_PARAMS["Agglomerative"].get("linkage", "ward"))
        link_combo = ttk.Combobox(self.hyperparams_frame, textvariable=self.agg_linkage_var, width=10,
                                  values=["ward", "complete", "average", "single"])
        link_combo.grid(row=row, column=1, padx=2)
        if self.agg_linkage_var.get() != "ward":
            row += 1
            tk.Label(self.hyperparams_frame, text="metric:").grid(row=row, column=0, padx=2)
            self.agg_metric_var = tk.StringVar(value=DEFAULT_PARAMS["Agglomerative"].get("metric", "euclidean"))
            met_combo = ttk.Combobox(self.hyperparams_frame, textvariable=self.agg_metric_var, width=10,
                                     values=["euclidean", "l1", "l2", "manhattan", "cosine"])
            met_combo.grid(row=row, column=1, padx=2)

    # ---------- Hyperparameter UI Builders for Dimensionality Reduction ----------
    def _build_pca_ui(self):
        row = 0
        tk.Label(self.hyperparams_frame, text="n_components:").grid(row=row, column=0, padx=2)
        self.pca_components_var = tk.IntVar(value=DEFAULT_PARAMS["PCA"].get("n_components", 2))
        tk.Entry(self.hyperparams_frame, textvariable=self.pca_components_var, width=5).grid(row=row, column=1, padx=2)
        row += 1
        tk.Label(self.hyperparams_frame, text="whiten:").grid(row=row, column=0, padx=2)
        self.pca_whiten_var = tk.BooleanVar(value=DEFAULT_PARAMS["PCA"].get("whiten", False))
        tk.Checkbutton(self.hyperparams_frame, variable=self.pca_whiten_var).grid(row=row, column=1, padx=2)

    def _build_tsne_ui(self):
        row = 0
        tk.Label(self.hyperparams_frame, text="Perplexity:").grid(row=row, column=0, padx=2)
        self.tsne_perp_var = tk.DoubleVar(value=DEFAULT_PARAMS["t-SNE"].get("perplexity", 30.0))
        tk.Entry(self.hyperparams_frame, textvariable=self.tsne_perp_var, width=5).grid(row=row, column=1, padx=2)
        row += 1
        tk.Label(self.hyperparams_frame, text="n_iter:").grid(row=row, column=0, padx=2)
        self.tsne_iter_var = tk.IntVar(value=DEFAULT_PARAMS["t-SNE"].get("n_iter", 1000))
        tk.Entry(self.hyperparams_frame, textvariable=self.tsne_iter_var, width=6).grid(row=row, column=1, padx=2)
        row += 1
        tk.Label(self.hyperparams_frame, text="learning_rate:").grid(row=row, column=0, padx=2)
        self.tsne_lr_var = tk.DoubleVar(value=DEFAULT_PARAMS["t-SNE"].get("learning_rate", 200.0))
        tk.Entry(self.hyperparams_frame, textvariable=self.tsne_lr_var, width=6).grid(row=row, column=1, padx=2)

    # ---------- Hyperparameter UI Builders for Anomaly Detection ----------
    def _build_isolationforest_ui(self):
        row = 0
        tk.Label(self.hyperparams_frame, text="n_estimators:").grid(row=row, column=0, padx=2)
        self.if_n_estimators_var = tk.IntVar(value=DEFAULT_PARAMS["Isolation Forest"].get("n_estimators", 100))
        tk.Entry(self.hyperparams_frame, textvariable=self.if_n_estimators_var, width=5).grid(row=row, column=1, padx=2)
        row += 1
        tk.Label(self.hyperparams_frame, text="contamination:").grid(row=row, column=0, padx=2)
        self.if_contamination_var = tk.DoubleVar(value=DEFAULT_PARAMS["Isolation Forest"].get("contamination", 0.1))
        tk.Entry(self.hyperparams_frame, textvariable=self.if_contamination_var, width=5).grid(row=row, column=1, padx=2)

    def _build_lof_ui(self):
        row = 0
        tk.Label(self.hyperparams_frame, text="n_neighbors:").grid(row=row, column=0, padx=2)
        self.lof_n_neighbors_var = tk.IntVar(value=DEFAULT_PARAMS["Local Outlier Factor"].get("n_neighbors", 20))
        tk.Entry(self.hyperparams_frame, textvariable=self.lof_n_neighbors_var, width=5).grid(row=row, column=1, padx=2)
        row += 1
        tk.Label(self.hyperparams_frame, text="contamination:").grid(row=row, column=0, padx=2)
        self.lof_contamination_var = tk.DoubleVar(value=DEFAULT_PARAMS["Local Outlier Factor"].get("contamination", 0.1))
        tk.Entry(self.hyperparams_frame, textvariable=self.lof_contamination_var, width=5).grid(row=row, column=1, padx=2)

    # ----------------------- DATA LOADING -----------------------
    def load_builtin_dataset(self):
        self.clear_memo2()
        task = self.task_var.get()
        ds_name = self.dataset_var.get()
        self.log_memo2(f"Loading built-in dataset: {ds_name}\n")
        datasets = DATASETS.get(task, [])
        for name, loader in datasets:
            if name == ds_name:
                try:
                    data = loader()
                    if task in ["Classification", "Regression"]:
                        self.X, self.y = data
                    else:
                        self.X = data
                        self.y = None
                    self.scaler = MinMaxScaler()
                    self.X = self.scaler.fit_transform(self.X)
                    self.log_memo2(f"Loaded dataset '{ds_name}' with shape {self.X.shape}\n")
                except Exception as e:
                    self.log_memo2(f"Error loading dataset: {e}\n")
                return
        self.log_memo2("Dataset not found.\n")

    def load_csv(self):
        self.clear_memo2()
        path = filedialog.askopenfilename(title="Select CSV", filetypes=[("CSV Files","*.csv"),("All","*.*")])
        if not path:
            return
        try:
            df_temp = pd.read_csv(path)
            df_temp.dropna(how="all", inplace=True)
            self.df = df_temp.copy()
            self.X = self.df.values
            self.y = None
            self.scaler = MinMaxScaler()
            self.X = self.scaler.fit_transform(self.X)
            self.log_memo2(f"Loaded user CSV: {path}\nShape: {self.df.shape}\n")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def show_data_info(self):
        if self.df is None:
            messagebox.showinfo("Info", "No dataset loaded.")
            return
        self.clear_memo2()
        info = f"DATA INFO:\nShape: {self.df.shape}\nColumns: {list(self.df.columns)}\n\n"
        info += str(self.df.describe(include='all'))
        self.log_memo2(info + "\n")

    def open_in_excel(self):
        if self.df is None:
            messagebox.showinfo("Info", "No dataset loaded.")
            return
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            temp_path = tmp.name
        self.df.to_csv(temp_path, index=False)
        try:
            os.startfile(temp_path)
            self.log_memo2(f"Opening dataset in Excel: {temp_path}\n")
        except:
            self.log_memo2("Could not open in Excel automatically.\nFile saved at: %s\n" % temp_path)

    # ----------------------- MODEL FIT -----------------------
    def fit_model(self):
        if self.X is None:
            self.log_memo2("No dataset loaded. Please load a dataset first.\n")
            return
        task = self.task_var.get()
        algo = self.algo_var.get()
        self.log_memo2(f"Fitting model. Task={task}, Algo={algo}\n")
        params = {}
        if task in ["Classification", "Regression"]:
            if algo in ["Random Forest", "Random Forest Regressor"]:
                params["n_estimators"] = self.rf_n_estimators_var.get()
                params["max_depth"] = self.rf_max_depth_var.get()
                params["min_samples_split"] = self.rf_min_split_var.get()
            elif algo in ["XGBoost", "XGBoost Regressor"]:
                params["n_estimators"] = self.xgb_n_estimators_var.get()
                params["learning_rate"] = self.xgb_lr_var.get()
                params["max_depth"] = self.xgb_max_depth_var.get()
            elif algo in ["AdaBoost", "AdaBoost Regressor"]:
                params["n_estimators"] = self.ab_n_estimators_var.get()
                params["learning_rate"] = self.ab_lr_var.get()
        elif task == "Clustering":
            if algo == "K-Means":
                params["n_clusters"] = self.km_n_clusters_var.get()
                params["init"] = self.km_init_var.get()
                params["max_iter"] = self.km_max_iter_var.get()
                params["n_init"] = self.km_n_init_var.get()
            elif algo == "DBSCAN":
                params["eps"] = self.dbscan_eps_var.get()
                params["min_samples"] = self.dbscan_min_var.get()
                params["metric"] = self.dbscan_metric_var.get()
                params["leaf_size"] = self.dbscan_leaf_var.get()
            elif algo == "Agglomerative":
                params["n_clusters"] = self.agg_n_clusters_var.get()
                params["linkage"] = self.agg_linkage_var.get()
                if self.agg_linkage_var.get() != "ward":
                    params["metric"] = self.agg_metric_var.get()
        elif task == "Dimensionality Reduction":
            if algo == "PCA":
                params["n_components"] = self.pca_components_var.get()
                params["whiten"] = self.pca_whiten_var.get()
            elif algo == "t-SNE":
                params["perplexity"] = self.tsne_perp_var.get()
                params["n_iter"] = self.tsne_iter_var.get()
                params["learning_rate"] = self.tsne_lr_var.get()
        elif task == "Anomaly Detection":
            if algo == "Isolation Forest":
                params["n_estimators"] = self.if_n_estimators_var.get()
                params["contamination"] = self.if_contamination_var.get()
            elif algo == "Local Outlier Factor":
                params["n_neighbors"] = self.lof_n_neighbors_var.get()
                params["contamination"] = self.lof_contamination_var.get()

        try:
            self.model_core = EnsembleDemoCore(task, algo, params, self.X, self.y)
            self.log_memo2("Model fitted successfully.\n")
            self.update_qtable_display()
        except Exception as e:
            self.log_memo2(f"Error in fitting model: {e}\n")

    # ----------------------- TEST SAMPLE -----------------------
    def test_sample(self):
        if self.model_core is None:
            messagebox.showinfo("Info", "No model fitted yet.")
            return
        inp = simpledialog.askstring("Test Sample", "Enter feature values (comma separated):")
        if not inp:
            return
        try:
            sample = np.array([float(x) for x in inp.split(',')]).reshape(1, -1)
            sample_scaled = self.scaler.transform(sample) if self.scaler else sample
            pred = self.model_core.predict_sample(sample_scaled)
            self.log_memo2(f"Test sample prediction: {pred}\n")
        except Exception as e:
            self.log_memo2(f"Error testing sample: {e}\n")

    # ----------------------- VISUALIZE -----------------------
    def visualize_model(self):
        if self.X is None:
            messagebox.showinfo("Info", "No dataset loaded.")
            return
        if self.model_core is None:
            messagebox.showinfo("Info", "No model fitted.")
            return
        task = self.task_var.get()
        algo = self.algo_var.get()
        X_plot = self.X.copy()
        import matplotlib.pyplot as plt
        fig = plt.Figure(figsize=(6,5), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_title(f"{task} - {algo}")
        if task in ["Classification", "Regression"]:
            ax.axis('off')
            ax.text(0.5, 0.5, f"Ensemble model ({algo}) fitted.\nUse test sample to see predictions.",
                    horizontalalignment='center', verticalalignment='center', fontsize=12)
        elif task == "Clustering":
            if X_plot.shape[1] > 2:
                from sklearn.decomposition import PCA
                X_plot = PCA(n_components=2, random_state=42).fit_transform(X_plot)
            if hasattr(self.model_core.model, "labels_"):
                labels = self.model_core.model.labels_
                sc = ax.scatter(X_plot[:,0], X_plot[:,1], c=labels, cmap="viridis")
                fig.colorbar(sc, ax=ax)
            else:
                ax.scatter(X_plot[:,0], X_plot[:,1], c='blue', alpha=0.7)
            ax.set_xlabel("Dim 1")
            ax.set_ylabel("Dim 2")
        elif task == "Dimensionality Reduction":
            if algo == "PCA":
                coords = self.model_core.model.transform(X_plot)
            elif algo == "t-SNE":
                coords = self.model_core.embedding_
            ax.scatter(coords[:,0], coords[:,1], c='blue', alpha=0.7)
            ax.set_xlabel("Dim 1")
            ax.set_ylabel("Dim 2")
        elif task == "Anomaly Detection":
            if algo == "Isolation Forest":
                pred = self.model_core.model.predict(X_plot)
                sc = ax.scatter(X_plot[:,0], X_plot[:,1], c=pred, cmap="coolwarm", alpha=0.7)
                fig.colorbar(sc, ax=ax)
            elif algo == "Local Outlier Factor":
                pred = self.model_core.lof_labels if hasattr(self.model_core, "lof_labels") else None
                if pred is not None:
                    sc = ax.scatter(X_plot[:,0], X_plot[:,1], c=pred, cmap="coolwarm", alpha=0.7)
                    fig.colorbar(sc, ax=ax)
                else:
                    ax.scatter(X_plot[:,0], X_plot[:,1], c='blue', alpha=0.7)
            ax.set_xlabel("Dim 1")
            ax.set_ylabel("Dim 2")
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        plot_win = tk.Toplevel(self.root)
        plot_win.title(f"Visualization - {task}/{algo}")
        canvas = FigureCanvasTkAgg(fig, master=plot_win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.clear_memo2()
        self.log_memo2("Visualization displayed.\n")

    # ----------------------- MODEL OUTPUT TEXT -----------------------
    def update_qtable_display(self):
        self.qtable_text.delete("1.0", tk.END)
        task = self.task_var.get()
        algo = self.algo_var.get()
        if task in ["Classification", "Regression"]:
            try:
                if algo in ["Random Forest", "Random Forest Regressor"]:
                    imp = self.model_core.model.feature_importances_
                    s = "Feature Importances:\n"
                    for i, v in enumerate(imp):
                        s += f"Feature {i}: {v:.4f}\n"
                    self.qtable_text.insert(tk.END, s)
                elif algo in ["XGBoost", "XGBoost Regressor"]:
                    if hasattr(self.model_core.model, "feature_importances_"):
                        imp = self.model_core.model.feature_importances_
                        s = "Feature Importances:\n"
                        for i, v in enumerate(imp):
                            s += f"Feature {i}: {v:.4f}\n"
                        self.qtable_text.insert(tk.END, s)
                    else:
                        self.qtable_text.insert(tk.END, "No feature importances available.\n")
                elif algo in ["AdaBoost", "AdaBoost Regressor"]:
                    if hasattr(self.model_core.model, "feature_importances_"):
                        imp = self.model_core.model.feature_importances_
                        s = "Feature Importances:\n"
                        for i, v in enumerate(imp):
                            s += f"Feature {i}: {v:.4f}\n"
                        self.qtable_text.insert(tk.END, s)
                    else:
                        self.qtable_text.insert(tk.END, "No feature importances available.\n")
                else:
                    self.qtable_text.insert(tk.END, "Model output not available for this task.\n")
            except Exception as e:
                self.qtable_text.insert(tk.END, f"Error displaying model output: {e}\n")
        else:
            self.qtable_text.insert(tk.END, f"Model output for {task} using {algo}:\n")
            if task == "Clustering":
                if hasattr(self.model_core.model, "labels_"):
                    self.qtable_text.insert(tk.END, f"Cluster labels: {np.unique(self.model_core.model.labels_)}\n")
                else:
                    self.qtable_text.insert(tk.END, "No cluster labels available.\n")
            elif task == "Dimensionality Reduction":
                if algo == "PCA":
                    self.qtable_text.insert(tk.END, f"Explained variance ratio: {self.model_core.model.explained_variance_ratio_}\n")
                elif algo == "t-SNE":
                    self.qtable_text.insert(tk.END, f"t-SNE embedding shape: {self.model_core.embedding_.shape}\n")
            elif task == "Anomaly Detection":
                if algo == "Isolation Forest":
                    self.qtable_text.insert(tk.END, "Isolation Forest anomaly predictions available via predict.\n")
                elif algo == "Local Outlier Factor":
                    self.qtable_text.insert(tk.END, "LOF anomaly labels computed.\n")
        self.qtable_text.see(tk.END)

    # ----------------------- TRAINING CONTROL -----------------------
    def on_start(self):
        if self.training_thread and self.training_thread.is_alive():
            self.log_memo2("Training thread already running.")
            return
        self.log_memo2("Starting ensemble model training (if iterative).")
        self.running = True
        self.paused = False
        self.step_mode = False
        self.training_thread = threading.Thread(target=self.training_loop, daemon=True)
        self.training_thread.start()

    def on_stop(self):
        self.log_memo2("Stopping training.")
        self.running = False

    def on_pause(self):
        if not self.running:
            self.log_memo2("Not running; cannot pause.")
            return
        self.log_memo2("Pausing training.")
        self.paused = True

    def on_resume(self):
        if not self.running:
            self.log_memo2("Not running; start first.")
            return
        if not self.paused:
            self.log_memo2("Already running; no need to resume.")
            return
        self.log_memo2("Resuming training.")
        self.paused = False

    def on_step(self):
        if not self.running:
            self.log_memo2("Starting in step mode.")
            self.running = True
            self.step_mode = True
            self.paused = False
            self.training_thread = threading.Thread(target=self.training_loop, daemon=True)
            self.training_thread.start()
        else:
            self.log_memo2("Performing one step.")
            self.step_mode = True
            self.paused = False

    def training_loop(self):
        iteration = 0
        STEPS_PER_ITER = 5
        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue
            iteration += 1
            # For demonstration, we refit the ensemble model at each iteration
            self.fit_model()
            self.log_memo2(f"Iteration {iteration}: Model fitted.")
            self.update_qtable_display()
            if self.step_mode:
                self.paused = True
                self.step_mode = False
            time.sleep(self.speed_var.get())
        self.log_memo2("Training loop ended.")

    def log_memo2(self, msg):
        self.memo2_text.insert(tk.END, msg + "\n")
        self.memo2_text.see(tk.END)

    def clear_memo2(self):
        self.memo2_text.delete("1.0", tk.END)

    # ----------------------- MAIN -----------------------
    def test_sample(self):
        if self.model_core is None:
            messagebox.showinfo("Info", "No model fitted yet.")
            return
        inp = simpledialog.askstring("Test Sample", "Enter feature values (comma separated):")
        if not inp:
            return
        try:
            sample = np.array([float(x) for x in inp.split(',')]).reshape(1, -1)
            sample_scaled = self.scaler.transform(sample) if self.scaler else sample
            pred = self.model_core.predict_sample(sample_scaled)
            self.log_memo2(f"Test sample prediction: {pred}\n")
        except Exception as e:
            self.log_memo2(f"Error testing sample: {e}\n")

    def visualize_model(self):
        if self.X is None:
            messagebox.showinfo("Info", "No dataset loaded.")
            return
        if self.model_core is None:
            messagebox.showinfo("Info", "No model fitted.")
            return
        task = self.task_var.get()
        algo = self.algo_var.get()
        X_plot = self.X.copy()
        import matplotlib.pyplot as plt
        fig = plt.Figure(figsize=(6,5), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_title(f"{task} - {algo}")
        if task in ["Classification", "Regression"]:
            ax.axis('off')
            ax.text(0.5, 0.5, f"Ensemble model ({algo}) fitted.\nUse test sample to see predictions.",
                    horizontalalignment='center', verticalalignment='center', fontsize=12)
        elif task == "Clustering":
            if X_plot.shape[1] > 2:
                from sklearn.decomposition import PCA
                X_plot = PCA(n_components=2, random_state=42).fit_transform(X_plot)
            if hasattr(self.model_core.model, "labels_"):
                labels = self.model_core.model.labels_
                sc = ax.scatter(X_plot[:,0], X_plot[:,1], c=labels, cmap="viridis")
                fig.colorbar(sc, ax=ax)
            else:
                ax.scatter(X_plot[:,0], X_plot[:,1], c='blue', alpha=0.7)
            ax.set_xlabel("Dim 1")
            ax.set_ylabel("Dim 2")
        elif task == "Dimensionality Reduction":
            if algo == "PCA":
                coords = self.model_core.model.transform(X_plot)
            elif algo == "t-SNE":
                coords = self.model_core.embedding_
            ax.scatter(coords[:,0], coords[:,1], c='blue', alpha=0.7)
            ax.set_xlabel("Dim 1")
            ax.set_ylabel("Dim 2")
        elif task == "Anomaly Detection":
            if algo == "Isolation Forest":
                pred = self.model_core.model.predict(X_plot)
                sc = ax.scatter(X_plot[:,0], X_plot[:,1], c=pred, cmap="coolwarm", alpha=0.7)
                fig.colorbar(sc, ax=ax)
            elif algo == "Local Outlier Factor":
                pred = self.model_core.lof_labels if hasattr(self.model_core, "lof_labels") else None
                if pred is not None:
                    sc = ax.scatter(X_plot[:,0], X_plot[:,1], c=pred, cmap="coolwarm", alpha=0.7)
                    fig.colorbar(sc, ax=ax)
                else:
                    ax.scatter(X_plot[:,0], X_plot[:,1], c='blue', alpha=0.7)
            ax.set_xlabel("Dim 1")
            ax.set_ylabel("Dim 2")
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        plot_win = tk.Toplevel(self.root)
        plot_win.title(f"Visualization - {task}/{algo}")
        canvas = FigureCanvasTkAgg(fig, master=plot_win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.clear_memo2()
        self.log_memo2("Visualization displayed.\n")

    def update_qtable_display(self):
        self.qtable_text.delete("1.0", tk.END)
        task = self.task_var.get()
        algo = self.algo_var.get()
        if task in ["Classification", "Regression"]:
            try:
                if algo in ["Random Forest", "Random Forest Regressor"]:
                    imp = self.model_core.model.feature_importances_
                    s = "Feature Importances:\n"
                    for i, v in enumerate(imp):
                        s += f"Feature {i}: {v:.4f}\n"
                    self.qtable_text.insert(tk.END, s)
                elif algo in ["XGBoost", "XGBoost Regressor"]:
                    if hasattr(self.model_core.model, "feature_importances_"):
                        imp = self.model_core.model.feature_importances_
                        s = "Feature Importances:\n"
                        for i, v in enumerate(imp):
                            s += f"Feature {i}: {v:.4f}\n"
                        self.qtable_text.insert(tk.END, s)
                    else:
                        self.qtable_text.insert(tk.END, "No feature importances available.\n")
                elif algo in ["AdaBoost", "AdaBoost Regressor"]:
                    if hasattr(self.model_core.model, "feature_importances_"):
                        imp = self.model_core.model.feature_importances_
                        s = "Feature Importances:\n"
                        for i, v in enumerate(imp):
                            s += f"Feature {i}: {v:.4f}\n"
                        self.qtable_text.insert(tk.END, s)
                    else:
                        self.qtable_text.insert(tk.END, "No feature importances available.\n")
                else:
                    self.qtable_text.insert(tk.END, "Model output not available for this task.\n")
            except Exception as e:
                self.qtable_text.insert(tk.END, f"Error displaying model output: {e}\n")
        else:
            self.qtable_text.insert(tk.END, f"Model output for {task} using {algo}:\n")
            if task == "Clustering":
                if hasattr(self.model_core.model, "labels_"):
                    self.qtable_text.insert(tk.END, f"Cluster labels: {np.unique(self.model_core.model.labels_)}\n")
                else:
                    self.qtable_text.insert(tk.END, "No cluster labels available.\n")
            elif task == "Dimensionality Reduction":
                if algo == "PCA":
                    self.qtable_text.insert(tk.END, f"Explained variance ratio: {self.model_core.model.explained_variance_ratio_}\n")
                elif algo == "t-SNE":
                    self.qtable_text.insert(tk.END, f"t-SNE embedding shape: {self.model_core.embedding_.shape}\n")
            elif task == "Anomaly Detection":
                if algo == "Isolation Forest":
                    self.qtable_text.insert(tk.END, "Isolation Forest anomaly predictions available via predict.\n")
                elif algo == "Local Outlier Factor":
                    self.qtable_text.insert(tk.END, "LOF anomaly labels computed.\n")
        self.qtable_text.see(tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = EnsembleDemoApp(root)
    root.mainloop()

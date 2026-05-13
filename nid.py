import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import threading
import os

class KDDIntrusionDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("🌐 KDD Intrusion Detection System")
        self.root.geometry("1600x1000")
        self.root.state('zoomed')
        
        self.df = None
        self.best_model = None
        self.results = {}
        
        self.setup_styles()
        self.setup_ui()
    
    def setup_styles(self):
        """Clean, modern styling that works everywhere"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure styles
        style.configure('Title.TLabel', font=('Segoe UI', 24, 'bold'))
        style.configure('Header.TLabel', font=('Segoe UI', 16, 'bold'))
        style.configure('Card.TFrame', relief='flat', padding=15)
        style.configure('Modern.TButton', font=('Segoe UI', 12, 'bold'), padding=10)
    
    def setup_ui(self):
        # Header
        header_frame = ttk.Frame(self.root, style='Card.TFrame')
        header_frame.pack(fill='x', padx=20, pady=(20,10))
        
        ttk.Label(header_frame, text="🛡️ KDD INTRUSION DETECTION", 
                 style='Title.TLabel').pack(pady=10)
        ttk.Label(header_frame, text="Random Forest Classifier Dashboard", 
                 font=('Segoe UI', 14)).pack()
        
        # Notebook
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Create tabs
        self.data_frame = ttk.Frame(self.notebook, style='Card.TFrame')
        self.notebook.add(self.data_frame, text="📊 Dataset")
        self.setup_data_tab()
        
        self.model_frame = ttk.Frame(self.notebook, style='Card.TFrame')
        self.notebook.add(self.model_frame, text="⚙️ Training")
        self.setup_model_tab()
        
        self.results_frame = ttk.Frame(self.notebook, style='Card.TFrame')
        self.notebook.add(self.results_frame, text="📈 Results")
        self.setup_results_tab()
    
    def setup_data_tab(self):
        # Upload section
        upload_frame = ttk.LabelFrame(self.data_frame, text="📁 Upload Dataset")
        upload_frame.pack(fill='x', pady=10)
        
        ttk.Button(upload_frame, text="📂 Choose KDD Dataset File", 
                  command=self.upload_file, style='Modern.TButton').pack(pady=20)
        
        self.file_label = ttk.Label(upload_frame, text="No file selected", 
                                   foreground='gray')
        self.file_label.pack(pady=5)
        
        # Stats
        stats_frame = ttk.LabelFrame(self.data_frame, text="📋 Dataset Statistics")
        stats_frame.pack(fill='x', pady=10)
        
        self.stats_labels = {}
        stats_grid = ttk.Frame(stats_frame)
        stats_grid.pack(pady=10)
        
        stats_items = [("Samples", "0"), ("Features", "0"), ("Normal %", "0%"), 
                      ("Attacks %", "0%"), ("Memory", "0 MB")]
        
        for i, (label, value) in enumerate(stats_items):
            frame = ttk.Frame(stats_grid)
            frame.grid(row=i//2, column=i%2, padx=20, pady=10, sticky='w')
            
            ttk.Label(frame, text=label+":").pack(anchor='w')
            self.stats_labels[label] = ttk.Label(frame, text=value, 
                                               font=('Segoe UI', 14, 'bold'))
            self.stats_labels[label].pack(anchor='w')
        
        # Sample data
        table_frame = ttk.LabelFrame(self.data_frame, text="👁️ Sample Data")
        table_frame.pack(fill='both', expand=True, pady=10)
        
        tree_frame = ttk.Frame(table_frame)
        tree_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.sample_tree = ttk.Treeview(tree_frame, height=15)
        vbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.sample_tree.yview)
        hbar = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.sample_tree.xview)
        
        self.sample_tree.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)
        self.sample_tree.pack(side='left', fill='both', expand=True)
        vbar.pack(side='right', fill='y')
        hbar.pack(side='bottom', fill='x')
    
    def setup_model_tab(self):
        params_frame = ttk.LabelFrame(self.model_frame, text="🎛️ Model Parameters")
        params_frame.pack(fill='x', pady=10)
        
        # Train size
        ttk.Label(params_frame, text="Training Size:", font=('Segoe UI', 12, 'bold')).pack(anchor='w')
        train_frame = ttk.Frame(params_frame)
        train_frame.pack(fill='x', pady=5)
        
        self.train_size = ttk.Scale(train_frame, from_=0.6, to=0.9, orient='horizontal')
        self.train_size.set(0.7)
        self.train_size.pack(side='left', fill='x', expand=True, padx=(0,10))
        self.train_size_label = ttk.Label(train_frame, text="70%", font=('Segoe UI', 16, 'bold'))
        self.train_size_label.pack(side='right')
        self.train_size.configure(command=self.update_train_label)
        
        # Parameters grid
        grid_frame = ttk.Frame(params_frame)
        grid_frame.pack(fill='x', pady=20)
        
        ttk.Label(grid_frame, text="Trees:").grid(row=0, column=0, sticky='w', padx=(0,10))
        self.n_trees = ttk.Combobox(grid_frame, values=['50', '100', '200', '300'], 
                                   state='readonly', width=10)
        self.n_trees.set('100')
        self.n_trees.grid(row=0, column=1, padx=10)
        
        ttk.Label(grid_frame, text="CV Folds:").grid(row=0, column=2, sticky='w', padx=(20,10))
        self.cv_folds = ttk.Combobox(grid_frame, values=['3', '5', '10'], 
                                    state='readonly', width=8)
        self.cv_folds.set('3')
        self.cv_folds.grid(row=0, column=3)
        
        # Train button
        self.train_btn = ttk.Button(params_frame, text="🚀 START TRAINING", 
                                   command=self.start_training, style='Modern.TButton')
        self.train_btn.pack(pady=30)
        
        # Progress
        self.progress = ttk.Progressbar(params_frame, mode='indeterminate', length=400)
        self.progress.pack(fill='x', pady=10)
        
        self.status_label = ttk.Label(params_frame, text="Ready to train...")
        self.status_label.pack(pady=5)
    
    def setup_results_tab(self):
        # Metrics
        metrics_frame = ttk.LabelFrame(self.results_frame, text="📊 Key Metrics")
        metrics_frame.pack(fill='x', pady=10)
        
        self.metric_frame = ttk.Frame(metrics_frame)
        self.metric_frame.pack(pady=10)
        
        # Charts
        charts_frame = ttk.LabelFrame(self.results_frame, text="📈 Visualizations")
        charts_frame.pack(fill='both', expand=True, pady=10)
        
        left_frame = ttk.Frame(charts_frame)
        left_frame.pack(side='left', fill='both', expand=True, padx=(0,10))
        ttk.Label(left_frame, text="Confusion Matrix", font=('Segoe UI', 12, 'bold')).pack()
        self.cm_frame = ttk.Frame(left_frame)
        self.cm_frame.pack(fill='both', expand=True)
        
        right_frame = ttk.Frame(charts_frame)
        right_frame.pack(side='right', fill='both', expand=True, padx=(10,0))
        ttk.Label(right_frame, text="Feature Importance", font=('Segoe UI', 12, 'bold')).pack()
        self.fi_frame = ttk.Frame(right_frame)
        self.fi_frame.pack(fill='both', expand=True)
        
        # Report
        report_frame = ttk.LabelFrame(self.results_frame, text="📋 Classification Report")
        report_frame.pack(fill='x', pady=10)
        self.report_text = tk.Text(report_frame, height=12, font=('Consolas', 10))
        scrollbar = ttk.Scrollbar(report_frame, orient='vertical', command=self.report_text.yview)
        self.report_text.configure(yscrollcommand=scrollbar.set)
        self.report_text.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
    
    def update_train_label(self, value):
        self.train_size_label.config(text=f"{float(value)*100:.0f}%")
    
    def upload_file(self):
        filename = filedialog.askopenfilename(
            title="Select KDD Dataset",
            filetypes=[("CSV/Text files", "*.csv *.txt"), ("All files", "*.*")]
        )
        if filename:
            try:
                self.df = pd.read_csv(filename, header=None)
                self.file_label.config(text=f"✅ Loaded: {os.path.basename(filename)}")
                
                normal_ratio = (self.df.iloc[:,-1].astype(str).str.lower() == 'normal').mean()
                
                self.stats_labels["Samples"].config(text=f"{self.df.shape[0]:,}")
                self.stats_labels["Features"].config(text=str(self.df.shape[1]-1))
                self.stats_labels["Normal %"].config(text=f"{normal_ratio:.1%}")
                self.stats_labels["Attacks %"].config(text=f"{1-normal_ratio:.1%}")
                self.stats_labels["Memory"].config(text=f"{self.df.memory_usage(deep=True).sum()/1024**2:.1f} MB")
                
                self.show_sample_data()
                messagebox.showinfo("Success", "Dataset loaded successfully!")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file:\n{str(e)}")
    
    def show_sample_data(self):
        for item in self.sample_tree.get_children():
            self.sample_tree.delete(item)
        
        if self.df is not None:
            cols_to_show = min(12, self.df.shape[1])
            self.sample_tree["columns"] = list(range(cols_to_show))
            self.sample_tree["show"] = "headings"
            
            for i, col in enumerate(self.sample_tree["columns"]):
                self.sample_tree.heading(col, text=f"F{i}")
                self.sample_tree.column(col, width=100)
            
            for _, row in self.df.head(15).iterrows():
                values = [str(x)[:12]+"..." if len(str(x))>12 else str(x) for x in row[:cols_to_show]]
                self.sample_tree.insert("", "end", values=values)
    
    def start_training(self):
        if self.df is None:
            messagebox.showwarning("Warning", "Please upload dataset first!")
            return
        
        self.train_btn.config(state='disabled')
        self.progress.start()
        self.status_label.config(text="🚀 Training model...")
        
        thread = threading.Thread(target=self.train_model, daemon=True)
        thread.start()
    
    def train_model(self):
        try:
            x = self.df.iloc[:, :-1].copy()
            y = self.df.iloc[:, -1].copy()
            
            # Preprocessing
            imputer = SimpleImputer(strategy='most_frequent')
            x = pd.DataFrame(imputer.fit_transform(x), columns=x.columns)
            
            # Encode categoricals
            cat_cols = [col for col in x.columns if x[col].dtype == 'object']
            for col in cat_cols:
                le = LabelEncoder()
                x[col] = le.fit_transform(x[col].astype(str))
            
            y = y.apply(lambda v: 0 if str(v).lower().strip() == "normal" else 1)
            
            # Split
            train_size = float(self.train_size.get())
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, train_size=train_size, random_state=42, stratify=y
            )
            
            # Train
            n_estimators = int(self.n_trees.get())
            param_grid = {
                'n_estimators': [n_estimators//2, n_estimators],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            }
            
            rfc = RandomForestClassifier(random_state=42)
            cv = int(self.cv_folds.get())
            grid_search = GridSearchCV(rfc, param_grid, cv=cv, scoring='f1', n_jobs=-1)
            grid_search.fit(x_train, y_train)
            
            self.best_model = grid_search.best_estimator_
            prediction = self.best_model.predict(x_test)
            
            # Results
            self.results = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cm': confusion_matrix(y_test, prediction, labels=[0, 1]),
                'report': classification_report(y_test, prediction, labels=[0, 1],
                                             target_names=["Normal", "Attack"],
                                             output_dict=True, zero_division=0),
                'feature_importance': pd.DataFrame({
                    'feature': x.columns,
                    'importance': self.best_model.feature_importances_
                }).sort_values('importance', ascending=False)
            }
            
            self.root.after(0, self.display_results)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            self.root.after(0, self.training_complete)
    
    def display_results(self):
        self.show_metrics()
        self.plot_confusion_matrix()
        self.plot_feature_importance()
        self.show_classification_report()
    
    def show_metrics(self):
        for widget in self.metric_frame.winfo_children():
            widget.destroy()
        
        metrics = [
            ("F1 Score", f"{self.results['best_score']:.4f}"),
            ("Accuracy", f"{self.results['report']['accuracy']:.4f}"),
            ("Attack Recall", f"{self.results['report']['Attack']['recall']:.4f}")
        ]
        
        for i, (name, value) in enumerate(metrics):
            frame = ttk.LabelFrame(self.metric_frame, text=name)
            frame.grid(row=0, column=i, padx=10, pady=10)
            ttk.Label(frame, text=value, font=('Segoe UI', 18, 'bold')).pack(pady=10)
    
    def plot_confusion_matrix(self):
        for widget in self.cm_frame.winfo_children():
            widget.destroy()
        
        fig, ax = plt.subplots(figsize=(6, 5))
        cm = self.results['cm']
        im = ax.imshow(cm, cmap='Blues')
        
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Normal', 'Attack'])
        ax.set_yticklabels(['Normal', 'Attack'])
        ax.set_title('Confusion Matrix')
        
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i,j]), ha='center', va='center', fontweight='bold')
        
        canvas = FigureCanvasTkAgg(fig, self.cm_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def plot_feature_importance(self):
        for widget in self.fi_frame.winfo_children():
            widget.destroy()
        
        top_features = self.results['feature_importance'].head(15)
        fig, ax = plt.subplots(figsize=(8, 6))
        
        features = [str(f)[:20] + "..." if len(str(f)) > 20 else str(f) 
                   for f in top_features['feature']]
        values = top_features['importance'].values
        
        bars = ax.barh(range(len(values)), values)
        ax.set_yticks(range(len(values)))
        ax.set_yticklabels(features)
        ax.set_xlabel('Importance')
        ax.set_title('Top 15 Features')
        
        for i, (bar, v) in enumerate(zip(bars, values)):
            ax.text(v + 0.001, i, f'{v:.3f}', va='center')
        
        canvas = FigureCanvasTkAgg(fig, self.fi_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def show_classification_report(self):
        self.report_text.delete(1.0, tk.END)
        report_df = pd.DataFrame(self.results['report']).round(4).T
        report_text = report_df.to_string() + f"\n\nBest Params: {self.results['best_params']}"
        self.report_text.insert(1.0, report_text)
    
    def training_complete(self):
        self.progress.stop()
        self.train_btn.config(state='normal')
        self.status_label.config(text="✅ Training completed!")
        self.notebook.select(2)

def main():
    root = tk.Tk()
    app = KDDIntrusionDetector(root)
    root.mainloop()

if __name__ == "__main__":
    main()

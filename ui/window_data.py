"""
window_data.py
==============
Window 1 — "Full Option" Data Engineering.
"""

from __future__ import annotations

import os
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLabel, QFileDialog, QTabWidget, QGroupBox, QComboBox, 
    QDoubleSpinBox, QSpinBox, QCheckBox, QListWidget, QListWidgetItem,
    QProgressBar, QMessageBox, QScrollArea, QFrame, QSplitter, QHeaderView,
    QDialog, QFormLayout
)

from ui.data_table_view import DataPreviewTable
from utils.project_state import ProjectState
from workers.data_loader_worker import DataLoaderWorker

class DataWindow(QMainWindow):
    """
    Exhaustive Data Engineering Window.
    """

    def __init__(
        self,
        project_state: ProjectState,
        on_next=None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.state = project_state
        self._on_next_callback = on_next
        
        self.df = None
        self.profile_df = None
        self.pipeline = None
        
        # Robust thread management
        self._active_threads = []
        self._active_workers = []

        self._init_ui()

    def _init_ui(self):
        self.setWindowTitle("Neural Forge — Data Engineering Lab")
        self.setMinimumSize(1000, 750)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        # Header
        header = QLabel("NEURAL FORGE")
        header.setStyleSheet("font-size: 10pt; font-weight: 800; letter-spacing: 2px; color: #00A3FF;")
        layout.addWidget(header)
        
        title = QLabel("Data Engineering Lab")
        title.setStyleSheet("font-size: 22pt; font-weight: 700; margin-bottom: 8px;")
        layout.addWidget(title)

        # Tabs
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # ── Tab 1: Ingestion ──
        self.tab_ingest = QWidget()
        self._setup_ingest_tab()
        self.tabs.addTab(self.tab_ingest, "INGESTION")

        # ── Tab 2: Cleaning ──
        self.tab_clean = QWidget()
        self._setup_clean_tab()
        self.tabs.addTab(self.tab_clean, "CLEANING")

        # ── Tab 3: Engineering ──
        self.tab_engineer = QWidget()
        self._setup_engineer_tab()
        self.tabs.addTab(self.tab_engineer, "ENGINEERING")

        # ── Tab 4: Preprocessing ──
        self.tab_preprocess = QWidget()
        self._setup_preprocess_tab()
        self.tabs.addTab(self.tab_preprocess, "MODEL")

        # ── Tab 5: Export/Deploy ──
        self.tab_export = QWidget()
        self._setup_export_tab()
        self.tabs.addTab(self.tab_export, "DEPLOY")

        # Progress Bar & Status
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(4)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Bottom Bar
        bottom_bar = QHBoxLayout()
        self.lbl_status = QLabel("Ready")
        self.lbl_status.setStyleSheet("color: #8b949e;")
        bottom_bar.addWidget(self.lbl_status)
        bottom_bar.addStretch()
        
        self.btn_next = QPushButton("Proceed to Model Builder →")
        self.btn_next.setProperty("class", "primary")
        self.btn_next.setMinimumHeight(40)
        self.btn_next.setEnabled(False)
        self.btn_next.clicked.connect(self._on_next)
        bottom_bar.addWidget(self.btn_next)
        
        layout.addLayout(bottom_bar)

    # ── Tab Setup Methods ──────────────────────────────────────────────────

    def _setup_ingest_tab(self):
        layout = QVBoxLayout(self.tab_ingest)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Ingestion Panel
        group = QGroupBox("Data Ingestion")
        g_lay = QHBoxLayout(group)
        self.btn_load = QPushButton("📂 Load CSV / Parquet")
        self.btn_load.clicked.connect(self._load_data)
        g_lay.addWidget(self.btn_load)
        
        self.lbl_file_info = QLabel("No file loaded.")
        g_lay.addWidget(self.lbl_file_info, stretch=1)
        
        self.btn_corr = QPushButton("📊 Show Correlation Matrix")
        self.btn_corr.clicked.connect(self._show_correlation)
        g_lay.addWidget(self.btn_corr)
        
        layout.addWidget(group)

        # Profiling Panel
        prof_group = QGroupBox("Statistical Profile")
        p_lay = QVBoxLayout(prof_group)
        self.profile_table = DataPreviewTable()
        p_lay.addWidget(self.profile_table)
        layout.addWidget(prof_group, stretch=1)

    def _setup_clean_tab(self):
        layout = QVBoxLayout(self.tab_clean)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # NaN Handling
        nan_group = QGroupBox("NaN Management")
        n_lay = QHBoxLayout(nan_group)
        n_lay.addWidget(QLabel("Strategy:"))
        self.combo_nan = QComboBox()
        self.combo_nan.addItems(["drop", "mean", "median", "mode", "constant"])
        n_lay.addWidget(self.combo_nan)
        
        self.edit_nan_const = QLabel("Const:") # Placeholder or QLineEdit
        # For simplicity, let's just use a label and assume user might add more if needed
        
        n_lay.addStretch()
        layout.addWidget(nan_group)

        # Outliers
        out_group = QGroupBox("Outlier Detection & Treatment")
        o_lay = QVBoxLayout(out_group)
        
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Method:"))
        self.combo_out_method = QComboBox()
        self.combo_out_method.addItems(["iqr", "z-score"])
        row1.addWidget(self.combo_out_method)
        
        row1.addWidget(QLabel("Action:"))
        self.combo_out_action = QComboBox()
        self.combo_out_action.addItems(["clip", "remove"])
        row1.addWidget(self.combo_out_action)
        o_lay.addLayout(row1)
        
        o_lay.addWidget(QLabel("Select columns for outlier treatment:"))
        self.list_out_cols = QListWidget()
        o_lay.addWidget(self.list_out_cols)
        
        layout.addWidget(out_group)

        self.btn_apply_clean = QPushButton("🧹 Apply Cleaning & Outlier Treatment")
        self.btn_apply_clean.clicked.connect(self._apply_cleaning)
        layout.addWidget(self.btn_apply_clean)

    def _setup_engineer_tab(self):
        layout = QVBoxLayout(self.tab_engineer)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        # Power Transforms
        pt_group = QGroupBox("Power Transforms")
        pt_lay = QVBoxLayout(pt_group)
        pt_lay.addWidget(QLabel("Select columns for Power Transform:"))
        self.list_pt_cols = QListWidget()
        self.list_pt_cols.setMinimumHeight(100)
        pt_lay.addWidget(self.list_pt_cols)
        layout.addWidget(pt_group)

        # Time-Series
        ts_group = QGroupBox("Time-Series & Temporal")
        ts_lay = QVBoxLayout(ts_group)
        
        form_ts = QFormLayout()
        form_ts.setSpacing(10)
        self.spin_lags = QSpinBox()
        self.spin_lags.setRange(0, 10)
        form_ts.addRow("Lag Features (t-n):", self.spin_lags)
        ts_lay.addLayout(form_ts)
        
        ts_lay.addSpacing(5)
        ts_lay.addWidget(QLabel("Select columns for Lag/Cyclical:"))
        self.list_ts_cols = QListWidget()
        self.list_ts_cols.setMinimumHeight(100)
        ts_lay.addWidget(self.list_ts_cols)
        layout.addWidget(ts_group)

        # Feature Interaction
        fi_group = QGroupBox("Feature Interaction")
        fi_lay = QVBoxLayout(fi_group)
        
        row_fi = QHBoxLayout()
        row_fi.setSpacing(12)
        
        for lbl, combo, stretch in [("Col A", "combo_fi_col1", 2), ("Op", "combo_fi_op", 1), ("Col B", "combo_fi_col2", 2)]:
            v = QVBoxLayout()
            v.setSpacing(4)
            v.addWidget(QLabel(lbl))
            setattr(self, combo, QComboBox())
            v.addWidget(getattr(self, combo))
            row_fi.addLayout(v, stretch=stretch)
        
        self.combo_fi_op.addItems(["add", "sub", "mul", "div"])
        
        vbtn = QVBoxLayout()
        vbtn.setSpacing(4)
        vbtn.addWidget(QLabel(" "))
        self.btn_fi_add = QPushButton("Create")
        self.btn_fi_add.clicked.connect(self._create_interaction)
        vbtn.addWidget(self.btn_fi_add)
        row_fi.addLayout(vbtn, stretch=1)
        
        fi_lay.addLayout(row_fi)
        layout.addWidget(fi_group)

        self.btn_apply_engineer = QPushButton("🧪 Apply Engineering")
        self.btn_apply_engineer.setMinimumHeight(42)
        self.btn_apply_engineer.setProperty("class", "primary")
        self.btn_apply_engineer.clicked.connect(self._apply_engineering)
        layout.addWidget(self.btn_apply_engineer)
        layout.addStretch()

    def _setup_preprocess_tab(self):
        layout = QVBoxLayout(self.tab_preprocess)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        # Target Selection
        target_group = QGroupBox("Target Selection")
        t_lay = QFormLayout(target_group)
        t_lay.setSpacing(10)
        
        self.combo_target = QComboBox()
        t_lay.addRow("Target Column:", self.combo_target)
        
        self.combo_problem_type = QComboBox()
        self.combo_problem_type.addItems(["classification", "regression"])
        t_lay.addRow("Problem Type:", self.combo_problem_type)
        layout.addWidget(target_group)

        # Feature Selection
        feat_group = QGroupBox("Feature Selection & Scaling")
        f_lay = QVBoxLayout(feat_group)
        
        form_scale = QFormLayout()
        form_scale.setSpacing(10)
        self.combo_scaling = QComboBox()
        self.combo_scaling.addItems(["standard", "minmax"])
        
        row_s = QHBoxLayout()
        row_s.addWidget(self.combo_scaling, stretch=1)
        self.btn_leakage = QPushButton("🔍 Check Leakage")
        self.btn_leakage.clicked.connect(self._check_leakage)
        row_s.addWidget(self.btn_leakage)
        
        form_scale.addRow("Scaling:", row_s)
        f_lay.addLayout(form_scale)
        
        f_lay.addSpacing(5)
        f_lay.addWidget(QLabel("Select features to INCLUDE:"))
        self.list_features = QListWidget()
        self.list_features.setMinimumHeight(120)
        f_lay.addWidget(self.list_features)
        layout.addWidget(feat_group)

        # PCA & Validation
        row_extra = QHBoxLayout()
        row_extra.setSpacing(12)
        
        pca_group = QGroupBox("PCA")
        pca_lay = QVBoxLayout(pca_group)
        self.check_pca = QCheckBox("Enable PCA")
        
        form_pca = QFormLayout()
        self.spin_pca = QDoubleSpinBox()
        self.spin_pca.setRange(0.5, 0.99)
        self.spin_pca.setValue(0.95)
        form_pca.addRow("Variance:", self.spin_pca)
        
        pca_lay.addWidget(self.check_pca)
        pca_lay.addLayout(form_pca)
        row_extra.addWidget(pca_group, stretch=1)
        
        val_group = QGroupBox("Validation")
        val_lay = QVBoxLayout(val_group)
        val_lay.addSpacing(10)
        self.btn_validate = QPushButton("✔️ Run Sanity Checks")
        self.btn_validate.clicked.connect(self._run_validation)
        val_lay.addWidget(self.btn_validate)
        row_extra.addWidget(val_group, stretch=1)
        
        layout.addLayout(row_extra)

        self.btn_apply_preprocess = QPushButton("⚙️ Apply Preprocessing Pipeline")
        self.btn_apply_preprocess.setMinimumHeight(42)
        self.btn_apply_preprocess.setProperty("class", "primary")
        self.btn_apply_preprocess.clicked.connect(self._apply_preprocessing)
        layout.addWidget(self.btn_apply_preprocess)
        layout.addStretch()

    def _setup_export_tab(self):
        layout = QVBoxLayout(self.tab_export)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Splitting
        split_group = QGroupBox("Data Splitting Strategy")
        s_lay = QVBoxLayout(split_group)
        
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Method:"))
        self.combo_split_method = QComboBox()
        self.combo_split_method.addItems(["percentage", "kfold"])
        row1.addWidget(self.combo_split_method)
        
        self.lbl_split_val = QLabel("Test Ratio:")
        row1.addWidget(self.lbl_split_val)
        self.spin_split = QDoubleSpinBox()
        self.spin_split.setRange(0.05, 0.5)
        self.spin_split.setValue(0.2)
        row1.addWidget(self.spin_split)
        s_lay.addLayout(row1)
        
        self.check_stratify = QCheckBox("Stratified Split (Keep class proportions)")
        self.check_stratify.setChecked(True)
        s_lay.addWidget(self.check_stratify)
        
        layout.addWidget(split_group)

        # Imbalance & Weights
        imb_group = QGroupBox("Imbalance Management")
        i_lay = QVBoxLayout(imb_group)
        
        row_imb = QHBoxLayout()
        row_imb.addWidget(QLabel("Resampling:"))
        self.combo_resample = QComboBox()
        self.combo_resample.addItems(["None", "smote", "undersample"])
        row_imb.addWidget(self.combo_resample)
        i_lay.addLayout(row_imb)
        
        self.check_weights = QCheckBox("Calculate Class Weights for Loss Function")
        i_lay.addWidget(self.check_weights)
        
        layout.addWidget(imb_group)

        # Export Actions
        export_group = QGroupBox("Export & Finalize")
        e_lay = QHBoxLayout(export_group)
        
        self.btn_save_pipeline = QPushButton(" Save Pipeline (Pickle)")
        self.btn_save_pipeline.clicked.connect(self._save_pipeline)
        e_lay.addWidget(self.btn_save_pipeline)
        
        self.btn_export = QPushButton(" Generate DataLoader & Finalize")
        self.btn_export.setProperty("class", "primary")
        self.btn_export.clicked.connect(self._export_data)
        e_lay.addWidget(self.btn_export)
        
        layout.addWidget(export_group)
        layout.addStretch()

    # ── Logic & Worker Integration ──────────────────────────────────────────

    def _start_worker(self, task: str, **kwargs):
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(True)
        self.btn_next.setEnabled(False)
        self.lbl_status.setText(f"Processing: {task}...")

        thread = QThread()
        worker = DataLoaderWorker(task, **kwargs)
        worker.moveToThread(thread)
        
        # Keep references to avoid GC
        self._active_threads.append(thread)
        self._active_workers.append(worker)
        
        thread.started.connect(worker.run)
        worker.progress.connect(self.lbl_status.setText)
        worker.error.connect(self._on_worker_error)
        
        # Pass the task type to the finisher
        worker.finished.connect(lambda res, rep, t=task: self._on_worker_finished(res, rep, t))
        
        # Cleanup when done
        worker.finished.connect(lambda: self._cleanup_task(thread, worker))
        worker.error.connect(lambda: self._cleanup_task(thread, worker))
        
        thread.start()

    def _on_worker_error(self, msg: str):
        self.progress_bar.setVisible(False)
        self.lbl_status.setText("Error occurred.")
        QMessageBox.critical(self, "Worker Error", msg)

    def _on_worker_finished(self, result: Any, report: Dict, task: str):
        self.progress_bar.setVisible(False)
        self.lbl_status.setText("Ready")
        
        if task == "load":
            self.df = result
            self.lbl_file_info.setText(f"Loaded: {report['path']} ({len(self.df)} rows)")
            self.tabs.setTabText(0, "INGESTION  ✅")
            self._start_worker("profile", df=self.df)
        elif task == "profile":
            self.profile_df = result
            self.profile_table.set_dataframe(self.profile_df)
            self._update_column_lists()
        elif task == "clean":
            self.df = result
            self.tabs.setTabText(1, "CLEANING  ✅")
            QMessageBox.information(self, "Success", "Cleaning operation completed.")
            self._start_worker("profile", df=self.df)
        elif task == "engineer":
            self.df = result
            self.tabs.setTabText(2, "ENGINEERING  ✅")
            QMessageBox.information(self, "Success", "Engineering operation completed.")
            self._start_worker("profile", df=self.df)
        elif task == "preprocess":
            self.df = result
            self.pipeline = report["pipeline"]
            self.tabs.setTabText(3, "MODEL  ✅")
            QMessageBox.information(self, "Success", "Preprocessing pipeline applied successfully.")
            self._start_worker("profile", df=self.df)
        elif task == "split":
            self.state.dataframe = self.df
            self.tabs.setTabText(4, "DEPLOY  ✅")
            self.btn_next.setEnabled(True)
            QMessageBox.information(self, "Ready", "Data is prepared. You can now proceed to Model Builder.")
        elif task == "interaction":
            self.df = result
            QMessageBox.information(self, "Success", "Feature interaction created.")
            self._start_worker("profile", df=self.df)
        elif task == "correlation":
            self._display_correlation(result)
        elif task == "leakage":
            if result:
                QMessageBox.warning(self, "Leakage Alert", f"The following columns have suspiciously high correlation (>0.95) with the target: {result}. They may cause target leakage.")
            else:
                QMessageBox.information(self, "Safe", "No significant target leakage detected.")
        elif task == "validate":
            if report["success"]:
                QMessageBox.information(self, "Validation Passed", "All domain constraints were respected.")
            else:
                QMessageBox.warning(self, "Validation Failed", "\n".join(report["errors"]))

    def _cleanup_task(self, thread, worker):
        """Safely stop and remove thread/worker."""
        if thread in self._active_threads:
            self._active_threads.remove(thread)
        if worker in self._active_workers:
            self._active_workers.remove(worker)
            
        thread.quit()
        # We don't wait() here to avoid blocking UI, 
        # but deleteLater will handle it once the event loop returns.
        thread.deleteLater()
        worker.deleteLater()

    def _update_column_lists(self):
        cols = list(self.df.columns)
        self.combo_target.clear()
        self.combo_target.addItems(cols)
        
        self.combo_fi_col1.clear()
        self.combo_fi_col1.addItems(cols)
        self.combo_fi_col2.clear()
        self.combo_fi_col2.addItems(cols)
        
        # Lists
        for lst in [self.list_out_cols, self.list_pt_cols, self.list_ts_cols, self.list_features]:
            lst.clear()
            for col in cols:
                item = QListWidgetItem(col)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked)
                lst.addItem(item)

    # ── Action Handlers ─────────────────────────────────────────────────────

    def _load_data(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Data", "", "Data Files (*.csv *.parquet)")
        if path:
            self._start_worker("load", filepath=path)

    def _apply_cleaning(self):
        if self.df is None: return
        out_cols = [self.list_out_cols.item(i).text() for i in range(self.list_out_cols.count()) 
                    if self.list_out_cols.item(i).checkState() == Qt.CheckState.Checked]
        
        self._start_worker("clean", df=self.df, 
                           strategy=self.combo_nan.currentText(),
                           outlier_cols=out_cols,
                           outlier_method=self.combo_out_method.currentText(),
                           outlier_action=self.combo_out_action.currentText())

    def _apply_engineering(self):
        if self.df is None: return
        pt_cols = [self.list_pt_cols.item(i).text() for i in range(self.list_pt_cols.count()) 
                   if self.list_pt_cols.item(i).checkState() == Qt.CheckState.Checked]
        ts_cols = [self.list_ts_cols.item(i).text() for i in range(self.list_ts_cols.count()) 
                   if self.list_ts_cols.item(i).checkState() == Qt.CheckState.Checked]
        
        # For cyclical, we assume a default max_val or let user specify (simplified here)
        cyclical_cols = [(c, 24) for c in ts_cols] # Defaulting to 24 for hour-like
        
        self._start_worker("engineer", df=self.df, 
                           cyclical_cols=cyclical_cols,
                           lag_cols=ts_cols,
                           n_lags=self.spin_lags.value())

    def _apply_preprocessing(self):
        if self.df is None: return
        target = self.combo_target.currentText()
        excluded = [self.list_features.item(i).text() for i in range(self.list_features.count()) 
                    if self.list_features.item(i).checkState() == Qt.CheckState.Unchecked]
        
        config = {
            'scaling': self.combo_scaling.currentText(),
            'exclude_columns': excluded,
            'power_transform_columns': [], # Could be refined
            'pca_enabled': self.check_pca.isChecked(),
            'pca_components': self.spin_pca.value()
        }
        
        self._start_worker("preprocess", df=self.df, target=target, config=config)
    
    def _show_correlation(self):
        if self.df is None: return
        self._start_worker("correlation", df=self.df)

    def _display_correlation(self, corr_df: pd.DataFrame):
        dlg = QDialog(self)
        dlg.setWindowTitle("Correlation Matrix — Feature Discovery")
        dlg.resize(900, 650)
        
        # Apply global stylesheet to the dialog
        from ui.styles import DARK_QSS
        dlg.setStyleSheet(DARK_QSS)
        
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(20, 20, 20, 20)
        
        header = QLabel("Correlation Matrix (Pearson)")
        header.setStyleSheet("font-size: 14pt; font-weight: 700; color: #00A3FF; margin-bottom: 10px;")
        lay.addWidget(header)
        
        desc = QLabel("Values near 1.0 or -1.0 indicate strong redundancy.")
        desc.setStyleSheet("color: #94A3B8; margin-bottom: 10px;")
        lay.addWidget(desc)
        
        table = DataPreviewTable()
        table.set_dataframe(corr_df)
        lay.addWidget(table)
        
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dlg.accept)
        lay.addWidget(btn_close, alignment=Qt.AlignmentFlag.AlignRight)
        
        dlg.exec()

    def _create_interaction(self):
        if self.df is None: return
        self._start_worker("interaction", df=self.df, 
                           col1=self.combo_fi_col1.currentText(),
                           col2=self.combo_fi_col2.currentText(),
                           op=self.combo_fi_op.currentText())

    def _check_leakage(self):
        if self.df is None: return
        self._start_worker("leakage", df=self.df, target=self.combo_target.currentText())

    def _run_validation(self):
        if self.df is None: return
        # Simple example constraints - in a real app, these would be user-defined
        # For now we'll just check if numeric columns are non-negative as a sanity check
        constraints = []
        for col in self.df.select_dtypes(include=[np.number]).columns:
            # dummy rule: if 'price' or 'temp' or 'age' in name, check positive
            if any(x in col.lower() for x in ['price', 'temp', 'age', 'val']):
                constraints.append({'column': col, 'op': 'greater', 'val': -0.0001})
        
        if not constraints:
            QMessageBox.information(self, "No Rules", "No default sanity rules found for your columns. Define domain constraints to validate.")
            return
            
        self._start_worker("validate", df=self.df, constraints=constraints)

    def _save_pipeline(self):
        if not self.pipeline:
            QMessageBox.warning(self, "No Pipeline", "Apply preprocessing first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Pipeline", "data_pipeline.pkl", "Pickle Files (*.pkl)")
        if path:
            self.pipeline.save(path)
            QMessageBox.information(self, "Saved", f"Pipeline saved to {path}")

    def _export_data(self):
        if self.df is None: return
        target = self.combo_target.currentText()
        config = {
            'method': self.combo_split_method.currentText(),
            'test_size': self.spin_split.value(),
            'stratify': self.check_stratify.isChecked(),
            'resample': self.combo_resample.currentText().lower() if self.combo_resample.currentIndex() > 0 else None,
            'calculate_weights': self.check_weights.isChecked()
        }
        self._start_worker("split", df=self.df, target=target, config=config)

    def _on_next(self):
        if self._on_next_callback:
            # Final sync to state
            self.state.target_column = self.combo_target.currentText()
            self.state.problem_type = self.combo_problem_type.currentText()
            self._on_next_callback()

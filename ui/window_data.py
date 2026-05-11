from __future__ import annotations
from PyQt6.QtCore import Qt, QThread
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog,
    QGroupBox, QComboBox, QDoubleSpinBox, QSpinBox, QCheckBox, QListWidget,
    QListWidgetItem, QProgressBar, QMessageBox, QScrollArea, QFormLayout
)
from ui.data_table_view import DataPreviewTable
from utils.project_state import ProjectState
from workers.data_loader_worker import DataLoaderWorker

class DataWindow(QWidget):
    """
    Exhaustive Data Engineering Dashboard (Scrollable).
    """
    def __init__(self, project_state: ProjectState, on_next=None, parent=None) -> None:
        super().__init__(parent)
        self.state = project_state
        self._on_next_callback = on_next
        self.df = None
        self.profile_df = None
        self.pipeline = None
        self._active_threads = []
        self._active_workers = []
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Scroll Area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(24)

        # Header
        header_layout = QVBoxLayout()
        title = QLabel("Data Engineering Lab")
        title.setProperty("class", "PageTitle")
        subtitle = QLabel("Ingest, clean, and preprocess your dataset for deep learning.")
        subtitle.setProperty("class", "PageSubtitle")
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        layout.addLayout(header_layout)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(6)
        layout.addWidget(self.progress_bar)

        self.lbl_status = QLabel("Ready")
        self.lbl_status.setStyleSheet("color: #64748B; font-size: 9pt;")
        layout.addWidget(self.lbl_status)

        # Grid-like layout using QHBoxLayout for cards
        row1 = QHBoxLayout()
        row1.addWidget(self._build_ingestion_card())
        row1.addWidget(self._build_target_card())
        layout.addLayout(row1)

        layout.addWidget(self._build_data_preview_card())

        layout.addWidget(self._build_profile_card())

        row2 = QHBoxLayout()
        row2.addWidget(self._build_cleaning_card())
        row2.addWidget(self._build_scaling_card())
        layout.addLayout(row2)

        layout.addWidget(self._build_engineering_card())

        # Bottom Actions
        bottom_row = QHBoxLayout()
        bottom_row.addStretch()
        self.btn_apply_all = QPushButton("Apply Full Preprocessing Pipeline")
        self.btn_apply_all.setProperty("class", "primary")
        self.btn_apply_all.setMinimumSize(300, 44)
        self.btn_apply_all.clicked.connect(self._apply_preprocessing)
        bottom_row.addWidget(self.btn_apply_all)

        self.btn_next = QPushButton("Proceed to Model Builder →")
        self.btn_next.setMinimumSize(250, 44)
        self.btn_next.setEnabled(False)
        self.btn_next.clicked.connect(self._on_next)
        bottom_row.addWidget(self.btn_next)

        layout.addLayout(bottom_row)

        scroll.setWidget(container)
        main_layout.addWidget(scroll)

    # ── Cards ────────────────────────────────────────────────────────

    def _build_ingestion_card(self) -> QGroupBox:
        group = QGroupBox("1. Data Ingestion")
        lay = QVBoxLayout(group)

        btn_load = QPushButton("📂 Load CSV / Parquet")
        btn_load.clicked.connect(self._load_data)
        lay.addWidget(btn_load)

        self.lbl_file_info = QLabel("No file loaded.")
        self.lbl_file_info.setWordWrap(True)
        lay.addWidget(self.lbl_file_info)
        lay.addStretch()
        return group

    def _build_target_card(self) -> QGroupBox:
        group = QGroupBox("2. Problem Definition")
        lay = QFormLayout(group)
        lay.setSpacing(12)

        self.combo_target = QComboBox()
        lay.addRow("Target Column:", self.combo_target)

        self.combo_problem_type = QComboBox()
        self.combo_problem_type.addItems(["classification", "regression"])
        lay.addRow("Problem Type:", self.combo_problem_type)
        return group

    def _build_data_preview_card(self) -> QGroupBox:
        group = QGroupBox("Raw Data Preview")
        lay = QVBoxLayout(group)
        self.raw_data_table = DataPreviewTable()
        self.raw_data_table.setMinimumHeight(200)
        lay.addWidget(self.raw_data_table)
        return group

    def _build_profile_card(self) -> QGroupBox:
        group = QGroupBox("Dataset Profile")
        lay = QVBoxLayout(group)
        self.profile_table = DataPreviewTable()
        self.profile_table.setMinimumHeight(200)
        lay.addWidget(self.profile_table)
        return group

    def _build_cleaning_card(self) -> QGroupBox:
        group = QGroupBox("3. Cleaning & Outliers")
        lay = QVBoxLayout(group)

        form = QFormLayout()
        self.combo_nan = QComboBox()
        self.combo_nan.addItems(["drop", "mean", "median", "mode", "knn"])
        form.addRow("NaN Strategy:", self.combo_nan)

        self.combo_out_method = QComboBox()
        self.combo_out_method.addItems(["iqr", "z-score"])
        form.addRow("Outlier Method:", self.combo_out_method)

        self.combo_out_action = QComboBox()
        self.combo_out_action.addItems(["clip", "remove"])
        form.addRow("Outlier Action:", self.combo_out_action)
        lay.addLayout(form)

        lay.addWidget(QLabel("Apply to Columns:"))
        self.list_out_cols = QListWidget()
        self.list_out_cols.setMaximumHeight(100)
        lay.addWidget(self.list_out_cols)

        btn_clean = QPushButton("Apply Cleaning")
        btn_clean.clicked.connect(self._apply_cleaning)
        lay.addWidget(btn_clean)
        return group

    def _build_scaling_card(self) -> QGroupBox:
        group = QGroupBox("4. Features & Scaling")
        lay = QVBoxLayout(group)

        form = QFormLayout()
        self.combo_scaling = QComboBox()
        self.combo_scaling.addItems(["standard", "minmax"])
        form.addRow("Scaling Method:", self.combo_scaling)

        self.check_pca = QCheckBox("Enable PCA")
        self.spin_pca = QDoubleSpinBox()
        self.spin_pca.setRange(0.5, 0.99)
        self.spin_pca.setValue(0.95)
        form.addRow(self.check_pca, self.spin_pca)
        lay.addLayout(form)

        lay.addWidget(QLabel("Include Features:"))
        self.list_features = QListWidget()
        self.list_features.setMaximumHeight(100)
        lay.addWidget(self.list_features)
        return group

    def _build_engineering_card(self) -> QGroupBox:
        group = QGroupBox("5. Advanced Feature Engineering")
        lay = QHBoxLayout(group)

        # Interactions
        fi_lay = QFormLayout()
        self.combo_fi_col1 = QComboBox()
        self.combo_fi_op = QComboBox()
        self.combo_fi_op.addItems(["add", "sub", "mul", "div"])
        self.combo_fi_col2 = QComboBox()

        fi_lay.addRow("Col A:", self.combo_fi_col1)
        fi_lay.addRow("Operator:", self.combo_fi_op)
        fi_lay.addRow("Col B:", self.combo_fi_col2)

        btn_fi = QPushButton("Create Interaction")
        btn_fi.clicked.connect(self._create_interaction)
        fi_lay.addRow("", btn_fi)

        lay.addLayout(fi_lay)
        lay.addSpacing(20)

        # Time Series
        ts_lay = QVBoxLayout()
        form_ts = QFormLayout()
        self.spin_lags = QSpinBox()
        self.spin_lags.setRange(0, 10)
        form_ts.addRow("Lags (t-n):", self.spin_lags)
        ts_lay.addLayout(form_ts)

        ts_lay.addWidget(QLabel("Lag Columns:"))
        self.list_ts_cols = QListWidget()
        self.list_ts_cols.setMaximumHeight(80)
        ts_lay.addWidget(self.list_ts_cols)

        btn_ts = QPushButton("Apply Lags")
        btn_ts.clicked.connect(self._apply_engineering)
        ts_lay.addWidget(btn_ts)

        lay.addLayout(ts_lay)
        lay.addSpacing(20)

        # Datetime Parsing
        dt_lay = QVBoxLayout()
        dt_lay.addWidget(QLabel("Date/Time Columns:"))
        self.list_dt_cols = QListWidget()
        self.list_dt_cols.setMaximumHeight(80)
        dt_lay.addWidget(self.list_dt_cols)

        btn_dt = QPushButton("Extract Date Features")
        btn_dt.clicked.connect(self._apply_datetime)
        dt_lay.addWidget(btn_dt)

        lay.addLayout(dt_lay)
        return group


    # ── Logic ────────────────────────────────────────────────────────

    def _start_worker(self, task: str, **kwargs):
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(True)
        self.btn_next.setEnabled(False)
        self.lbl_status.setText(f"Processing: {task}...")

        thread = QThread()
        worker = DataLoaderWorker(task, **kwargs)
        worker.moveToThread(thread)

        self._active_threads.append(thread)
        self._active_workers.append(worker)

        thread.started.connect(worker.run)
        worker.progress.connect(self.lbl_status.setText)
        worker.error.connect(self._on_worker_error)
        worker.finished.connect(lambda res, rep, t=task: self._on_worker_finished(res, rep, t))
        worker.finished.connect(lambda: self._cleanup_task(thread, worker))
        worker.error.connect(lambda: self._cleanup_task(thread, worker))
        thread.start()

    def _on_worker_error(self, msg: str):
        self.progress_bar.setVisible(False)
        self.lbl_status.setText("Error")
        QMessageBox.critical(self, "Error", msg)

    def _on_worker_finished(self, result, report, task: str):
        self.progress_bar.setVisible(False)
        self.lbl_status.setText("Ready")

        if task == "load":
            self.df = result
            self.lbl_file_info.setText(f"Loaded: {report['path']}\nRows: {len(self.df)}")
            self.raw_data_table.set_dataframe(self.df.head(100))
            self._start_worker("profile", df=self.df)
        elif task == "profile":
            self.profile_df = result
            self.profile_table.set_dataframe(self.profile_df)
            self._update_column_lists()
            if self.state.pipeline is not None:
                self.btn_next.setEnabled(True)
        elif task == "clean":
            self.df = result
            self.raw_data_table.set_dataframe(self.df.head(100))
            self._start_worker("profile", df=self.df)
        elif task == "engineer":
            self.df = result
            self.raw_data_table.set_dataframe(self.df.head(100))
            self._start_worker("profile", df=self.df)
        elif task == "interaction":
            self.df = result
            self.raw_data_table.set_dataframe(self.df.head(100))
            self._start_worker("profile", df=self.df)
        elif task == "preprocess":
            self.df = result
            self.raw_data_table.set_dataframe(self.df.head(100))
            self.state.dataframe = self.df
            self.state.target_column = self.combo_target.currentText()
            self.state.problem_type = self.combo_problem_type.currentText()
            self.state.pipeline = report.get("pipeline")
            self.btn_next.setEnabled(True)
            QMessageBox.information(self, "Success", "Data preprocessing applied! You can now proceed to Model Builder.")
            self._start_worker("profile", df=self.df)

    def _cleanup_task(self, thread, worker):
        if thread in self._active_threads: self._active_threads.remove(thread)
        if worker in self._active_workers: self._active_workers.remove(worker)
        thread.quit()
        thread.deleteLater()
        worker.deleteLater()

    def _update_column_lists(self):
        cols = list(self.df.columns)
        previous_target = self.state.target_column or self.combo_target.currentText()
        self.combo_target.clear()
        self.combo_target.addItems(cols)
        if previous_target in cols:
            self.combo_target.setCurrentText(previous_target)
        elif cols:
            self.combo_target.setCurrentIndex(len(cols) - 1)
        self.combo_fi_col1.clear()
        self.combo_fi_col1.addItems(cols)
        self.combo_fi_col2.clear()
        self.combo_fi_col2.addItems(cols)

        for lst in [self.list_out_cols, self.list_ts_cols, self.list_features, self.list_dt_cols]:
            lst.clear()
            for col in cols:
                item = QListWidgetItem(col)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked)
                lst.addItem(item)

    # ── Actions ──

    def _load_data(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Data", "", "Data Files (*.csv *.parquet)")
        if path:
            self._start_worker("load", filepath=path)

    def _apply_cleaning(self):
        if self.df is None: return
        out_cols = [self.list_out_cols.item(i).text() for i in range(self.list_out_cols.count()) if self.list_out_cols.item(i).checkState() == Qt.CheckState.Checked]
        self._start_worker("clean", df=self.df, strategy=self.combo_nan.currentText(), outlier_cols=out_cols, outlier_method=self.combo_out_method.currentText(), outlier_action=self.combo_out_action.currentText())

    def _create_interaction(self):
        if self.df is None: return
        self._start_worker("interaction", df=self.df, col1=self.combo_fi_col1.currentText(), col2=self.combo_fi_col2.currentText(), op=self.combo_fi_op.currentText())

    def _apply_engineering(self):
        if self.df is None: return
        ts_cols = [self.list_ts_cols.item(i).text() for i in range(self.list_ts_cols.count()) if self.list_ts_cols.item(i).checkState() == Qt.CheckState.Checked]
        self._start_worker("engineer", df=self.df, cyclical_cols=[], lag_cols=ts_cols, n_lags=self.spin_lags.value())

    def _apply_datetime(self):
        if self.df is None: return
        dt_cols = [self.list_dt_cols.item(i).text() for i in range(self.list_dt_cols.count()) if self.list_dt_cols.item(i).checkState() == Qt.CheckState.Checked]
        if not dt_cols: return
        self._start_worker("engineer", df=self.df, datetime_cols=dt_cols)

    def _apply_preprocessing(self):
        if self.df is None: return
        target = self.combo_target.currentText()
        excluded = [self.list_features.item(i).text() for i in range(self.list_features.count()) if self.list_features.item(i).checkState() == Qt.CheckState.Unchecked]
        config = {
            'scaling': self.combo_scaling.currentText(),
            'exclude_columns': excluded,
            'power_transform_columns': [],
            'pca_enabled': self.check_pca.isChecked(),
            'pca_components': self.spin_pca.value()
        }
        self._start_worker("preprocess", df=self.df, target=target, config=config)

    def _on_next(self):
        self.state.target_column = self.combo_target.currentText()
        self.state.problem_type = self.combo_problem_type.currentText()
        if self._on_next_callback:
            self._on_next_callback()

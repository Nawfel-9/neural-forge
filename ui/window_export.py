from __future__ import annotations
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QGroupBox, QMessageBox
)
from PyQt6.QtCore import Qt
import torch

from utils.project_state import ProjectState
from backend.exporter import export_to_onnx

class ExportWindow(QWidget):
    """
    Dedicated Export Dashboard.
    Handles saving the PyTorch model, ONNX model, and preprocessing pipeline.
    """
    def __init__(self, project_state: ProjectState, parent=None):
        super().__init__(parent)
        self.state = project_state
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 32, 32, 32)
        root.setSpacing(24)

        # ── Header ──
        header_layout = QVBoxLayout()
        title = QLabel("Deployment & Export")
        title.setProperty("class", "PageTitle")
        subtitle = QLabel("Export your trained models and preprocessing pipelines for production.")
        subtitle.setProperty("class", "PageSubtitle")
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        root.addLayout(header_layout)

        # ── Status Panel ──
        status_group = QGroupBox("Project Status")
        status_lay = QVBoxLayout(status_group)
        self.lbl_pipeline_status = QLabel("❌ Preprocessing Pipeline: Not Ready")
        self.lbl_model_status = QLabel("❌ Trained Model: Not Ready")

        for lbl in [self.lbl_pipeline_status, self.lbl_model_status]:
            lbl.setStyleSheet("font-weight: 600; font-size: 11pt;")
            status_lay.addWidget(lbl)

        btn_refresh = QPushButton("🔄 Refresh Status")
        btn_refresh.setMinimumHeight(36)
        btn_refresh.clicked.connect(self.refresh_status)
        status_lay.addWidget(btn_refresh, alignment=Qt.AlignmentFlag.AlignRight)

        root.addWidget(status_group)

        # ── Export Options ──
        export_row = QHBoxLayout()

        # 1. Pipeline Export
        pipe_group = QGroupBox("Data Pipeline")
        pipe_lay = QVBoxLayout(pipe_group)
        pipe_desc = QLabel("Export the data engineering steps (scaling, encoding, etc.) to apply on new data.")
        pipe_desc.setWordWrap(True)
        pipe_desc.setStyleSheet("color: #64748B; margin-bottom: 12px;")
        pipe_lay.addWidget(pipe_desc)

        self.btn_export_pipeline = QPushButton("💾 Save Pipeline (.pkl)")
        self.btn_export_pipeline.setMinimumHeight(44)
        self.btn_export_pipeline.clicked.connect(self._export_pipeline)
        pipe_lay.addWidget(self.btn_export_pipeline)
        pipe_lay.addStretch()
        export_row.addWidget(pipe_group)

        # 2. PyTorch Export
        pt_group = QGroupBox("PyTorch Model")
        pt_lay = QVBoxLayout(pt_group)
        pt_desc = QLabel("Export the native PyTorch model state dictionary for continued training or native inference.")
        pt_desc.setWordWrap(True)
        pt_desc.setStyleSheet("color: #64748B; margin-bottom: 12px;")
        pt_lay.addWidget(pt_desc)

        self.btn_export_pt = QPushButton("💾 Save PyTorch Model (.pt)")
        self.btn_export_pt.setMinimumHeight(44)
        self.btn_export_pt.clicked.connect(self._export_pt)
        pt_lay.addWidget(self.btn_export_pt)
        pt_lay.addStretch()
        export_row.addWidget(pt_group)

        # 3. ONNX Export
        onnx_group = QGroupBox("ONNX Interoperability")
        onnx_lay = QVBoxLayout(onnx_group)
        onnx_desc = QLabel("Export to Open Neural Network Exchange format for cross-platform inference (C++, JS, etc.).")
        onnx_desc.setWordWrap(True)
        onnx_desc.setStyleSheet("color: #64748B; margin-bottom: 12px;")
        onnx_lay.addWidget(onnx_desc)

        self.btn_export_onnx = QPushButton("📦 Export ONNX (.onnx)")
        self.btn_export_onnx.setProperty("class", "primary")
        self.btn_export_onnx.setMinimumHeight(44)
        self.btn_export_onnx.clicked.connect(self._export_onnx)
        onnx_lay.addWidget(self.btn_export_onnx)
        onnx_lay.addStretch()
        export_row.addWidget(onnx_group)

        root.addLayout(export_row)
        root.addStretch()

        self.refresh_status()

    def refresh_status(self):
        has_pipe = getattr(self.state, "pipeline", None) is not None
        has_model = self.state.model is not None and self.state.dummy_tensor is not None

        if has_pipe:
            self.lbl_pipeline_status.setText("✅ Preprocessing Pipeline: Ready")
            self.lbl_pipeline_status.setStyleSheet("color: #10B981; font-weight: bold; font-size: 11pt;")
            self.btn_export_pipeline.setEnabled(True)
        else:
            self.lbl_pipeline_status.setText("❌ Preprocessing Pipeline: Not Ready")
            self.lbl_pipeline_status.setStyleSheet("color: #EF4444; font-weight: bold; font-size: 11pt;")
            self.btn_export_pipeline.setEnabled(False)

        if has_model:
            self.lbl_model_status.setText("✅ Trained Model: Ready")
            self.lbl_model_status.setStyleSheet("color: #10B981; font-weight: bold; font-size: 11pt;")
            self.btn_export_pt.setEnabled(True)
            self.btn_export_onnx.setEnabled(True)
        else:
            self.lbl_model_status.setText("❌ Trained Model: Not Ready")
            self.lbl_model_status.setStyleSheet("color: #EF4444; font-weight: bold; font-size: 11pt;")
            self.btn_export_pt.setEnabled(False)
            self.btn_export_onnx.setEnabled(False)

    def _export_pipeline(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Pipeline", "data_pipeline.pkl", "Pickle Files (*.pkl)")
        if path:
            try:
                self.state.pipeline.save(path)
                QMessageBox.information(self, "Success", f"Pipeline saved to:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Failed", str(e))

    def _export_pt(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save PyTorch Model", "model.pt", "PyTorch Files (*.pt *.pth)")
        if path:
            try:
                torch.save(self.state.model.state_dict(), path)
                QMessageBox.information(self, "Success", f"Model state_dict saved to:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Failed", str(e))

    def _export_onnx(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save ONNX Model", "model.onnx", "ONNX Models (*.onnx)")
        if path:
            try:
                success, msg = export_to_onnx(self.state.model, self.state.dummy_tensor, path)
                if success:
                    QMessageBox.information(self, "Export Success", msg)
                else:
                    QMessageBox.critical(self, "Export Failed", msg)
            except Exception as exc:
                QMessageBox.critical(self, "Export Failed", f"Unexpected error:\n{exc}")

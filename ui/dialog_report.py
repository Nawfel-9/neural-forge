from __future__ import annotations
import datetime
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QTextEdit, QFileDialog, QMessageBox
)
from PyQt6.QtGui import QTextDocument, QFont
from PyQt6.QtPrintSupport import QPrinter

from utils.project_state import ProjectState

class ReportDialog(QDialog):
    """
    Professional Project Synthesis Dialog.
    Compiles data, architecture, and training results into a clean report.
    """
    def __init__(self, state: ProjectState, parent=None):
        super().__init__(parent)
        self.state = state
        self.setWindowTitle("Project Synthesis Report")
        self.resize(800, 700)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        title = QLabel("Final Project Report")
        title.setStyleSheet("font-size: 18pt; font-weight: 800; color: #0EA5E9;")
        layout.addWidget(title)

        self.report_view = QTextEdit()
        self.report_view.setReadOnly(True)
        # Professional styling for the preview
        self.report_view.setStyleSheet("""
            QTextEdit {
                background-color: #0F172A;
                color: #F1F5F9;
                border: 1px solid #1E293B;
                border-radius: 8px;
                padding: 12px;
                font-family: 'Segoe UI', sans-serif;
                font-size: 10pt;
            }
        """)
        self._generate_report_content()
        layout.addWidget(self.report_view)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        
        self.btn_pdf = QPushButton("📕 Export PDF")
        self.btn_pdf.setMinimumHeight(40)
        self.btn_pdf.setFixedWidth(140)
        self.btn_pdf.clicked.connect(self._export_pdf)
        btn_row.addWidget(self.btn_pdf)

        self.btn_close = QPushButton("Close")
        self.btn_close.setMinimumHeight(40)
        self.btn_close.setFixedWidth(100)
        self.btn_close.clicked.connect(self.close)
        btn_row.addWidget(self.btn_close)

        layout.addLayout(btn_row)

    def _generate_report_content(self, theme='dark'):
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Theme colors
        if theme == 'dark':
            bg_color = "#0F172A"
            text_color = "#F1F5F9"
            muted_color = "#94A3B8"
            header_color = "#F8FAFC"
            card_bg = "#1E293B"
            accent_blue = "#0EA5E9"
            sub_blue = "#38BDF8"
            border_color = "#1E293B"
        else: # Light theme for PDF
            bg_color = "#FFFFFF"
            text_color = "#0F172A"
            muted_color = "#475569"
            header_color = "#0F172A"
            card_bg = "#F1F5F9"
            accent_blue = "#0284C7"
            sub_blue = "#0369A1"
            border_color = "#E2E8F0"

        # 1. Data Analysis Summary
        ds_stats = "N/A"
        target_info = "N/A"
        file_name = "None"
        feature_count = self.state.input_features()
        stats_table = ""
        
        if self.state.dataset_path:
            from pathlib import Path
            file_name = Path(self.state.dataset_path).name
        
        if self.state.dataframe is not None:
            rows, cols = self.state.dataframe.shape
            ds_stats = f"{rows:,} samples, {cols:,} total columns"
            target_info = f"'{self.state.target_column}' ({self.state.problem_type})"
            
            # Generate statistical summary for numerical columns
            numerical_df = self.state.dataframe.select_dtypes(include=['number'])
            if not numerical_df.empty:
                desc = numerical_df.describe().T
                stats_rows = ""
                for col_name, row in desc.head(10).iterrows():
                    stats_rows += f"""
                    <tr>
                        <td style='padding: 6px; border-bottom: 1px solid {border_color}; font-weight: 600;'>{col_name}</td>
                        <td style='padding: 6px; border-bottom: 1px solid {border_color};'>{row['mean']:.2f}</td>
                        <td style='padding: 6px; border-bottom: 1px solid {border_color};'>{row['std']:.2f}</td>
                        <td style='padding: 6px; border-bottom: 1px solid {border_color};'>{row['min']:.2f}</td>
                        <td style='padding: 6px; border-bottom: 1px solid {border_color};'>{row['max']:.2f}</td>
                    </tr>
                    """
                stats_table = f"""
                <div style='margin-top: 15px;'>
                    <b style='color: {muted_color}; font-size: 9pt;'>FEATURE DISTRIBUTION (TOP 10):</b>
                    <table style='width: 100%; border-collapse: collapse; margin-top: 5px; font-size: 8.5pt; color: {text_color};'>
                        <thead>
                            <tr style='background: {card_bg}; color: {muted_color};'>
                                <th style='text-align: left; padding: 6px;'>FEATURE</th>
                                <th style='text-align: left; padding: 6px;'>MEAN</th>
                                <th style='text-align: left; padding: 6px;'>STD</th>
                                <th style='text-align: left; padding: 6px;'>MIN</th>
                                <th style='text-align: left; padding: 6px;'>MAX</th>
                            </tr>
                        </thead>
                        <tbody>{stats_rows}</tbody>
                    </table>
                </div>
                """
        
        # Preprocessing Justifications
        prep_html = ""
        c_cfg = self.state.cleaning_config
        p_cfg = self.state.prep_config
        
        if c_cfg.get("nan_strategy") and c_cfg["nan_strategy"] != "none":
            method = c_cfg["nan_strategy"].upper()
            prep_html += f"""
            <div style='margin-bottom: 12px;'>
                <span style='background: {card_bg}; color: {sub_blue}; padding: 2px 6px; border-radius: 4px; font-weight: 600;'>Imputation: {method}</span>
                <p style='margin: 4px 0 0 0; font-size: 8.5pt; color: {muted_color};'>Maintains structural integrity by populating null values, preventing batch processing failures.</p>
            </div>
            """
            
        if p_cfg.get("scaling") and p_cfg["scaling"] != "none":
            method = p_cfg["scaling"].capitalize()
            prep_html += f"""
            <div style='margin-bottom: 12px;'>
                <span style='background: {card_bg}; color: {sub_blue}; padding: 2px 6px; border-radius: 4px; font-weight: 600;'>{method} Scaling</span>
                <p style='margin: 4px 0 0 0; font-size: 8.5pt; color: {muted_color};'>Normalizes features to ensure stable gradient descent and prevent dominant feature bias.</p>
            </div>
            """
            
        if self.state.pipeline and getattr(self.state.pipeline, "encoders", {}):
            prep_html += f"""
            <div style='margin-bottom: 12px;'>
                <span style='background: {card_bg}; color: {sub_blue}; padding: 2px 6px; border-radius: 4px; font-weight: 600;'>Categorical Encoding</span>
                <p style='margin: 4px 0 0 0; font-size: 8.5pt; color: {muted_color};'>Transforms text attributes into mathematical vector spaces compatible with tensor operations.</p>
            </div>
            """
            
        if p_cfg.get("pca_enabled"):
            comps = p_cfg.get("pca_components", "Auto")
            prep_html += f"""
            <div style='margin-bottom: 12px;'>
                <span style='background: {card_bg}; color: {sub_blue}; padding: 2px 6px; border-radius: 4px; font-weight: 600;'>PCA (Reduction to {comps})</span>
                <p style='margin: 4px 0 0 0; font-size: 8.5pt; color: {muted_color};'>Reduces dimensionality while preserving maximum variance, helping to combat the "curse of dimensionality".</p>
            </div>
            """

        if not prep_html:
            prep_html = f"<p style='color: {muted_color}; font-style: italic;'>No preprocessing operations were applied to the raw data.</p>"

        # Architecture Audit
        layers_rows = ""
        if self.state.blueprint:
            for i, layer in enumerate(self.state.blueprint):
                l_type = layer.get('type', 'Unknown').upper()
                l_units = layer.get('units', 'N/A')
                l_act = layer.get('activation', 'None')
                layers_rows += f"""
                <tr>
                    <td style='padding: 8px; border-bottom: 1px solid {border_color}; color: {text_color};'>{i+1}</td>
                    <td style='padding: 8px; border-bottom: 1px solid {border_color}; font-weight: bold; color: {text_color};'>{l_type}</td>
                    <td style='padding: 8px; border-bottom: 1px solid {border_color}; color: {text_color};'>{l_units}</td>
                    <td style='padding: 8px; border-bottom: 1px solid {border_color}; color: #A855F7;'>{l_act}</td>
                </tr>
                """
        else:
            layers_rows = f"<tr><td colspan='4' style='padding: 20px; text-align: center; color: {muted_color};'>No layers defined in blueprint</td></tr>"
        
        # 3. Training Hyperparameters
        hp = self.state.hyperparams
        hp_table = f"""
        <table style='width: 100%; border-collapse: collapse; margin-top: 10px; color: {text_color};'>
            <tr>
                <td style='width: 25%; color: {muted_color};'>Learning Rate</td><td style='font-weight: 600;'>{hp.get('lr', 0.001)}</td>
                <td style='width: 25%; color: {muted_color};'>Batch Size</td><td style='font-weight: 600;'>{hp.get('batch_size', 32)}</td>
            </tr>
            <tr>
                <td style='color: {muted_color};'>Optimizer</td><td style='font-weight: 600;'>{self.state.optimizer_name}</td>
                <td style='color: {muted_color};'>Loss Function</td><td style='font-weight: 600;'>{self.state.loss_fn_name}</td>
            </tr>
            <tr>
                <td style='color: {muted_color};'>Epochs</td><td style='font-weight: 600;'>{hp.get('epochs', 50)}</td>
                <td style='color: {muted_color};'>Compute Device</td><td style='font-weight: 600; color: #10B981;'>{self.state.device.upper()}</td>
            </tr>
        </table>
        """

        # 4. Performance Metrics
        metrics_html = ""
        if self.state.training_metrics:
            metrics_html = "<div style='margin-top: 10px;'>"
            for k, v in self.state.training_metrics.items():
                metrics_html += f"""
                <div style='background: {card_bg}; border: 1px solid {border_color}; padding: 10px; border-radius: 8px; margin-bottom: 5px;'>
                    <span style='font-size: 8pt; color: #10B981; font-weight: 700; text-transform: uppercase;'>{k}: </span>
                    <span style='font-size: 12pt; font-weight: 800; color: {text_color};'>{v:.4f}</span>
                </div>
                """
            metrics_html += "</div>"
        else:
            metrics_html = f"<p style='color: {muted_color}; font-style: italic;'>No evaluation metrics found. Training may not have completed.</p>"

        # Final Build
        html = f"""
        <div style='font-family: sans-serif; color: {text_color}; background-color: {bg_color}; padding: 20px;'>
            <table style='width: 100%; margin-bottom: 20px;'>
                <tr>
                    <td>
                        <h1 style='color: {accent_blue}; margin: 0; font-size: 24pt;'>NEURAL FORGE</h1>
                        <div style='color: {sub_blue}; font-weight: 600; letter-spacing: 1px;'>EXPERT PROJECT SYNTHESIS</div>
                    </td>
                    <td style='text-align: right; vertical-align: middle;'>
                        <div style='color: {muted_color}; font-size: 9pt;'>REPORT ID: NF-{datetime.datetime.now().strftime("%Y%m%d%H%M")}</div>
                        <div style='color: {muted_color}; font-size: 9pt;'>DATE: {now}</div>
                    </td>
                </tr>
            </table>
            
            <div style='background: {border_color}; height: 2px; margin-bottom: 30px;'></div>

            <h2 style='color: {header_color}; border-left: 4px solid {accent_blue}; padding-left: 10px; font-size: 14pt;'>I. DATA ENGINEERING AUDIT</h2>
            <table style='width: 100%; margin-bottom: 20px; color: {text_color};'>
                <tr>
                    <td style='width: 50%;'><b>Source File:</b> <span style='color: {sub_blue};'>{file_name}</span></td>
                    <td><b>Dimensions:</b> {ds_stats}</td>
                </tr>
                <tr>
                    <td><b>Target Variable:</b> {target_info}</td>
                    <td><b>Input Features:</b> {feature_count}</td>
                </tr>
            </table>
            {stats_table}
            <div style='margin-bottom: 30px; margin-top: 25px;'>
                <b style='color: {header_color}; font-size: 10pt;'>PIPELINE OPERATIONS & JUSTIFICATION:</b>
                <div style='margin-top: 10px;'>
                    {prep_html}
                </div>
            </div>

            <h2 style='color: {header_color}; border-left: 4px solid #A855F7; padding-left: 10px; font-size: 14pt;'>II. NEURAL ARCHITECTURE</h2>
            <table style='width: 100%; border-collapse: collapse; margin-bottom: 30px;'>
                <thead>
                    <tr style='background: {card_bg};'>
                        <th style='text-align: left; padding: 10px; color: {muted_color};'>#</th>
                        <th style='text-align: left; padding: 10px; color: {muted_color};'>LAYER TYPE</th>
                        <th style='text-align: left; padding: 10px; color: {muted_color};'>UNITS/CONFIG</th>
                        <th style='text-align: left; padding: 10px; color: {muted_color};'>ACTIVATION</th>
                    </tr>
                </thead>
                <tbody>
                    {layers_rows}
                </tbody>
            </table>

            <h2 style='color: {header_color}; border-left: 4px solid #10B981; padding-left: 10px; font-size: 14pt;'>III. TRAINING CONFIGURATION</h2>
            {hp_table}
            
            <div style='margin-top: 30px; margin-bottom: 30px;'>
                <h2 style='color: {header_color}; border-left: 4px solid #F59E0B; padding-left: 10px; font-size: 14pt;'>IV. PERFORMANCE ANALYTICS</h2>
                {metrics_html}
            </div>

            <div style='background: {border_color}; height: 1px; margin-top: 40px;'></div>
            <p style='text-align: center; color: {muted_color}; font-size: 8pt; margin-top: 20px;'>
                CONFIDENTIAL TECHNICAL DOCUMENT - GENERATED BY NEURAL FORGE AI ENGINE v1.2<br/>
                All architecture designs and weights are property of the user environment.
            </p>
        </div>
        """
        if theme == 'dark':
            self.report_view.setHtml(html)
        return html

    def _export_pdf(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export PDF Report", "report.pdf", "PDF Files (*.pdf)")
        if not path:
            return

        try:
            # Create a dedicated document for printing with a light theme
            # (Dark mode text is invisible on standard white PDF backgrounds)
            print_doc = QTextDocument()
            
            # Use the dedicated light theme generator for PDF
            light_html = self._generate_report_content(theme='light')
            
            print_doc.setHtml(light_html)

            printer = QPrinter(QPrinter.PrinterMode.HighResolution)
            printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
            printer.setOutputFileName(path)
            
            # Correct margin setting for PyQt6
            from PyQt6.QtGui import QPageLayout, QPageSize
            from PyQt6.QtCore import QMarginsF
            
            page_layout = QPageLayout(
                QPageSize(QPageSize.PageSizeId.A4),
                QPageLayout.Orientation.Portrait,
                QMarginsF(15, 15, 15, 15)
            )
            printer.setPageLayout(page_layout)

            print_doc.print(printer)
            QMessageBox.information(self, "Success", f"Professional report exported successfully to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Could not generate PDF:\n{e}")

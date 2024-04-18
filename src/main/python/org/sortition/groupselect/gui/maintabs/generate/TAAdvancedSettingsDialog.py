from PyQt5.QtWidgets import QDialog, QPushButton, QLineEdit, QFormLayout, QLabel, QWidget, QHBoxLayout, QVBoxLayout
from PyQt5.QtGui import QIntValidator, QDoubleValidator

class TAAdvancedSettingsDialog(QDialog):
    def __init__(self, parent, swap_rounds_default, seed_default, pareto_default):
        super(TAAdvancedSettingsDialog, self).__init__(parent)

        self.ok = False

        self.swap_rounds_field = QLineEdit()
        self.swap_rounds_field.setValidator( QIntValidator(1, 100, self) );
        self.swap_rounds_field.setText(str(swap_rounds_default))

        self.seed_field = QLineEdit()
        self.seed_field.setValidator( QDoubleValidator() );
        self.seed_field.setText(str(seed_default))

        self.pareto_field = QLineEdit()
        self.pareto_field.setValidator( QDoubleValidator(0, 1, 2, self) );
        self.pareto_field.setText(str(pareto_default))

        form = QFormLayout()
        form.addRow(QLabel("Number of swap rounds:"), self.swap_rounds_field)
        form.addRow(QLabel("Random Number Seed:"), self.seed_field)
        form.addRow(QLabel("Prioritize demographic balance (high value) or number of meetings (low value)?"), self.pareto_field)
        form_widget = QWidget()
        form_widget.setLayout(form)

        self.btn_ok = QPushButton("Ok")
        self.btn_ok.clicked.connect(self.button_press)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.button_press)
        self.btn_cancel.move(80, 0)

        buttons = QHBoxLayout()
        buttons.addWidget(self.btn_ok)
        buttons.addWidget(self.btn_cancel)
        buttons_widget = QWidget()
        buttons_widget.setLayout(buttons)

        layout = QVBoxLayout()
        layout.addWidget(form_widget)
        layout.addWidget(buttons_widget)
        self.setLayout(layout)

    def button_press(self):
        if self.sender() == self.btn_ok:
            self.ok = True
        self.close()

    @classmethod
    def get_input(cls, parent, swap_rounds_default, seed_default, pareto_default):
        dialog = cls(parent, swap_rounds_default, seed_default, pareto_default)
        dialog.exec_()
        return (dialog.ok, float(dialog.swap_rounds_field.text()) if dialog.swap_rounds_field.text() else swap_rounds_default, float(dialog.seed_field.text()) if dialog.seed_field.text() else seed_default, float(dialog.pareto_field.text()) if dialog.pareto_field.text() else pareto_default)

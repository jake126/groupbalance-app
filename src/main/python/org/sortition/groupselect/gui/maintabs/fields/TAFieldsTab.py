from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QTableWidget, QLabel, QTableWidgetItem, QListWidget, QComboBox, QGroupBox, \
                            QGridLayout, QStackedLayout, QListWidgetItem, QHeaderView

class TAFieldsTab(QWidget):
    def __init__(self, ctx):
        super(TAFieldsTab, self).__init__()
        self.ctx = ctx

        self._table_being_updated = False

        self.create_ui()

    def create_ui(self):
        self.fields_list = QListWidget()
        self.fields_list.itemSelectionChanged.connect(self.userchanged_field_list)

        self.mode_group = QGroupBox("Field Mode")

        self.mode_box = QComboBox()
        self.mode_box.addItem("Ignore", 'ignore')
        self.mode_box.addItem("Print Label", 'print')
        self.mode_box.addItem("Cluster", 'cluster')
        self.mode_box.addItem("Diversify", 'diversify')
        self.mode_box.currentIndexChanged.connect(self.userchanged_mode_box)

        echoLayout = QGridLayout()
        echoLayout.addWidget(QLabel("Mode:"), 0, 0)
        echoLayout.addWidget(self.mode_box, 0, 1)
        echoLayout.setHorizontalSpacing(50)
        echoLayout.setColumnStretch(0,1)
        echoLayout.setColumnStretch(1,1)
        self.mode_group.setLayout(echoLayout)

        self.terms_group = QGroupBox("Field Values")
        self.terms_layout = QStackedLayout()
        self.terms_layout.addWidget(self.create_empty_term_widget())
        self.terms_layout.addWidget(self.create_table_term_widget())
        self.terms_group.setLayout(self.terms_layout)

        layout = QGridLayout()
        layout.addWidget(self.fields_list, 0, 0, 2, 1)
        layout.addWidget(self.mode_group, 0, 1, 1, 1)
        layout.addWidget(self.terms_group, 1, 1)
        layout.setRowStretch(1,1)
        self.setLayout(layout)

    def create_empty_term_widget(self):
        label = QLabel("Only applied for diversify and cluster categories.")
        label.setAlignment(Qt.AlignCenter)

        return label

    def create_table_term_widget(self):
        self.terms_table = QTableWidget(0, 3)
        self.terms_table.setHorizontalHeaderLabels(['Terms Found', 'Terms Usage', 'Cluster Value?'])
        self.terms_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.terms_table.cellChanged.connect(self.userchanged_table)

        return self.terms_table

    def display_none(self):
        self.fields_list.clearSelection()
        self.status_mode_box(False)
        self.status_terms_group(False)

    def get_current_field_index(self):
        return self.fields_list.currentItem().data(Qt.UserRole)

    def init_field(self, j):
        if j not in self.ctx.app_data.fields:
            mode = 'ignore'
            terms = [[t,t] for t in self.ctx.app_data_manager.get_terms(j)]
            self.ctx.app_data.fields[j] = {'mode': mode, 'terms': terms}
        else:
            for t in self.ctx.app_data_manager.get_terms(j):
                if not any(a[0] == t for a in self.ctx.app_data.fields[j]['terms']):
                    self.ctx.app_data.fields[j]['terms'].append([t,t])
            for k, term_usage in enumerate(self.ctx.app_data.fields[j]['terms']):
                if term_usage[0] not in self.ctx.app_data_manager.get_terms(j):
                    self.ctx.app_data.fields[j]['terms'].pop(k)

    def update_fields_list(self):
        self._field_list_being_updated = True
        
        self.fields_list.clear()
        for j, cat in enumerate(self.ctx.app_data.peopledata_keys):
            new_item = QListWidgetItem()
            new_item.setData(Qt.UserRole, j)
            new_item.setText(cat)
            self.fields_list.addItem(new_item)

        self._field_list_being_updated = False

    def update_mode_box(self, j):
        index = self.mode_box.findData(self.ctx.app_data.fields[j]['mode'])
        self.mode_box.setCurrentIndex(index)

    def update_terms_group(self, j):
        self._table_being_updated = True
        self.terms_table.setRowCount(len(self.ctx.app_data.fields[j]['terms']))
        for k, term_usage in enumerate(self.ctx.app_data.fields[j]['terms']):
            item_col0 = QTableWidgetItem(term_usage[0])
            item_col0.setFlags(Qt.ItemIsSelectable)
            self.terms_table.setItem(k, 0, item_col0)
            self.terms_table.setItem(k, 1, QTableWidgetItem(term_usage[1]))
            checkbox_item = QTableWidgetItem()
            checkbox_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            checkbox_item.setCheckState(Qt.Unchecked)
            self.terms_table.setItem(k, 2, checkbox_item)
        self._table_being_updated = False

    def status_mode_box(self, status):
        self.mode_group.setDisabled(not status)

    def status_terms_group(self, status):
        self.terms_group.setDisabled(not status)
        self.terms_layout.setCurrentIndex(1 if status else 0)

    def userchanged_field_list(self):
        if self._field_list_being_updated: return
        j = self.get_current_field_index()
        self.init_field(j)

        self.update_mode_box(j)
        self.update_terms_group(j)

        self.status_mode_box(True)
        mode = self.ctx.app_data.fields[j]['mode']
        self.status_terms_group(True if mode in ['cluster', 'diversify'] else False)

    def userchanged_mode_box(self, index):
        j = self.get_current_field_index()
        mode = self.mode_box.currentData()

        self.ctx.app_data.fields[j]['mode'] = mode

        mode = self.ctx.app_data.fields[j]['mode']
        if mode == 'diversify':
            self.terms_table.setColumnHidden(2, True)  # Hide the third column
        else:
            self.terms_table.setColumnHidden(2, False)  # Show the third column
        self.status_terms_group(True if mode in ['cluster', 'diversify'] else False)

        self.ctx.window.tabs.fields_update()

    def userchanged_table(self, k, l):
        if self._table_being_updated: return
        j = self.get_current_field_index()
        if l == 1:
            self.ctx.app_data.fields[j]['terms'][k][1] = self.terms_table.item(k, 1).text()
        if l == 2: 
            # ensure only one box can be checked
            for row in range(self.terms_table.rowCount()):
                if row != k:
                    item = self.terms_table.item(row, 2)
                    item.setCheckState(Qt.Unchecked)
            # Toggle checkbox state
            checkbox_item = self.terms_table.item(k, l)
            checked = checkbox_item.checkState() == Qt.Checked
            if checked:
                self.ctx.app_data.settings['val_cluster'] = self.terms_table.item(k, 1).text()
            if not checked:
                # reset to default
                self.ctx.app_data.settings['val_cluster'] = "cluster"
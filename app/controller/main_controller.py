from PyQt5.QtCore import QObject, pyqtSlot
from ..model.sqlite_model import ModelSQLite
from ..view.sql_view import MainWindow

class ControlerMain(QObject):
    def __init__(self, model:ModelSQLite, view:MainWindow) -> None:
        super().__init__()
        self.model = model
        self.view = view
        self.model.data_changed.connect(self.update_table)

    @pyqtSlot()
    def update_table(self):
        data = self.model.read_all()
        donor_data = self.model.read_donor_data()
        self.view.fill_table(data)
        self.view.fill_donor_table(donor_data)
    
    @pyqtSlot()
    def delete_row(self):
        id = self.view.ui.tableWidget.currentIndex().siblingAtColumn(0).data()
        self.model.delete_object(id)
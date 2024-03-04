import sys
from PyQt5.QtWidgets import QApplication
from app.model.sqlite_model import ModelSQLite
from app.controller.sql_controller import ControlerSQLite
from app.controller.add_controller import ControlerAdd
from app.view.sql_view import MainWindow, AddWindow, DeleteWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)

    model = ModelSQLite()

    main_controller = ControlerSQLite(model)
    add_controller = ControlerAdd(model, None)


    add_view = AddWindow(add_controller)
    add_controller.view = add_view

    main_view = MainWindow(main_controller)
    main_view.ui2 = add_view

    main_view.show()
    app.exec_()
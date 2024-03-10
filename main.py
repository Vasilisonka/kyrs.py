import sys
from PyQt5.QtWidgets import QApplication
from app.model.sqlite_model import ModelSQLite
from app.controller.main_controller import ControlerMain
from app.controller.add_controller import ControlerAdd
from app.view.sql_view import MainWindow, AddWindow, DeleteWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)

    model = ModelSQLite()

    main_controller = ControlerMain(model, None)
    add_controller = ControlerAdd(model, None)


    add_view = AddWindow(add_controller)
    add_controller.view = add_view

    main_view = MainWindow(main_controller)
    main_controller.view = main_view
    main_view.ui2 = add_view

    # initial data load
    main_controller.update_table()

    main_view.show()
    app.exec_()
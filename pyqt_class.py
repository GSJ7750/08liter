import sys
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
import pandas as pd

class MyQDialog(QtWidgets.QDialog):

    def __init__(self, product_name, data, parent=None):
        super(MyQDialog, self).__init__(parent)
        
        self.left_selected_items_list = list()
        self.right_selected_items_list = list()
        
        font = QtGui.QFont()
        font.setPointSize(18)
        fontMetrics = QtGui.QFontMetrics(font)
        
        
        frameStyle = QtWidgets.QFrame.Sunken | QtWidgets.QFrame.Panel
        
        #-------------------------------위젯 초기화
        self.pushButton = QtWidgets.QPushButton('제출')
        self.pushButton.setObjectName("pushButton")
        
        self.product_name_label = QtWidgets.QLabel('제품명')
        
        self.TextEdit = QtWidgets.QTextEdit()
        self.TextEdit.setFont(font)
        self.TextEdit.setObjectName("plainTextEdit")
        self.TextEdit.setPlainText(product_name)
        self.TextEdit.setReadOnly(True)
        textSize = fontMetrics.size(0, self.TextEdit.toPlainText())
        w = textSize.width() + 20
        h = textSize.height() + 10
        self.TextEdit.setMinimumSize(w, h)
        self.TextEdit.setMaximumSize(w, h)
        self.TextEdit.resize(w, h)
        self.TextEdit.setAlignment(QtCore.Qt.AlignCenter)
        
        
        self.label_left = QtWidgets.QLabel('남은 목록')
        self.label_right = QtWidgets.QLabel('버릴 목록')
        
        self.left_select_all_button = QtWidgets.QPushButton('전체선택')
        self.left_deselect_button = QtWidgets.QPushButton('선택해제')
        self.left_selected_cell_button = QtWidgets.QPushButton('선택된 셀 체크')
        
        self.right_select_all_button = QtWidgets.QPushButton('전체선택')
        self.right_deselect_button = QtWidgets.QPushButton('선택해제')
        self.right_selected_cell_button = QtWidgets.QPushButton('선택된 셀 체크')
        
        self.insert_button = QtWidgets.QPushButton('버리기\n>>>')
        self.remove_button = QtWidgets.QPushButton('꺼내기\n<<<')
        
        self.data = [[QtWidgets.QCheckBox(), idx, d] for idx, d in enumerate(data)]
        
        self.tmp_data = []
        
        self.init_tables()
        #-------------------------------
        
        
        #-------------------------------함수 연결
        self.left_select_all_button.clicked.connect(self.left_select_all)
        self.left_deselect_button.clicked.connect(self.left_deselect_all)
        self.left_selected_cell_button.clicked.connect(self.left_selected_cell_select)
        
        self.right_select_all_button.clicked.connect(self.right_select_all)
        self.right_deselect_button.clicked.connect(self.right_deselect_all)
        self.right_selected_cell_button.clicked.connect(self.right_selected_cell_select)
        
        self.pushButton.clicked.connect(self.close)
        self.insert_button.clicked.connect(self.insert)
        self.remove_button.clicked.connect(self.remove)
        
        #-------------------------------
        

        ##------------------------------ Set layout
        layout = QtWidgets.QGridLayout()
        inner_layout = QtWidgets.QGridLayout()
        left_inner_layout = QtWidgets.QGridLayout()
        right_inner_layout = QtWidgets.QGridLayout()
        #-------------------------------위젯 위치
        #---좌측
        layout.addWidget(self.product_name_label,0,0)
        layout.addWidget(self.TextEdit,1,0)
        left_inner_layout.addWidget(self.label_left, 0, 0)
        left_inner_layout.addWidget(self.left_select_all_button, 0, 1)
        left_inner_layout.addWidget(self.left_deselect_button, 0, 3)
        left_inner_layout.addWidget(self.left_selected_cell_button, 0, 4)
        layout.addLayout(left_inner_layout, 2, 0, QtCore.Qt.AlignLeft)
        layout.addWidget(self.tableWidget, 3, 0)
        
        
        #---우측
        right_inner_layout.addWidget(self.label_right, 0, 0)
        right_inner_layout.addWidget(self.right_select_all_button, 0, 1)
        right_inner_layout.addWidget(self.right_deselect_button, 0, 3)
        right_inner_layout.addWidget(self.right_selected_cell_button, 0, 4)
        layout.addLayout(right_inner_layout, 2, 2, QtCore.Qt.AlignLeft)
        layout.addWidget(self.tableWidget2, 3, 2)
        
        layout.addWidget(self.pushButton, 4, 2)
        
        
        #---가운데
        inner_layout.addWidget(self.insert_button, 2, 1)
        inner_layout.addWidget(self.remove_button, 3, 1)
        
        layout.addLayout(inner_layout, 3, 1, QtCore.Qt.AlignCenter)
        #-------------------------------
        
        self.setLayout(layout)
        self.setWindowTitle("Thank you for reading!")
        self.resize(1800,1012)
        
        
        
    #-------------------------------함수
    def left_selected_cell_select(self):
        items = self.tableWidget.selectedItems()
        n = 2
        items = [items[i:i + n] for i in range(0, len(items), n)]
        
        selected_items_list = list()
        for item in items:
            selected_items_list.append(int(item[0].text()))
        self.left_selected_items_list = sorted(selected_items_list)
        
        selected_list = self.left_selected_items_list
        for data in self.data:
            if data[1] in selected_list:
                data[0].setChecked(True)
        self.left_selected_items_list.clear()
        self.tableWidget.clearSelection()


    def right_selected_cell_select(self):
        items = self.tableWidget2.selectedItems()
        n = 2
        items = [items[i:i + n] for i in range(0, len(items), n)]
        
        selected_items_list = list()
        for item in items:
            selected_items_list.append(int(item[0].text()))
        self.right_selected_items_list = sorted(selected_items_list)
        
        selected_list = self.right_selected_items_list
        for data in self.tmp_data:
            if data[1] in selected_list:
                data[0].setChecked(True)
        self.right_selected_items_list.clear()
        self.tableWidget2.clearSelection()
        
    def init_tables(self):
        self.tableWidget = QtWidgets.QTableWidget(len(self.data),3)
        self.tableWidget.setObjectName("tableWidget")
        header = self.tableWidget.horizontalHeader()
        #header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.tableWidget.setColumnWidth(0,40)
        self.tableWidget.setColumnWidth(1,40)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        
        for cell_idx in range(len(self.data)):
            cellWidget = QtWidgets.QWidget()
            layoutCB = QtWidgets.QHBoxLayout(cellWidget)
            #self.data[cell_idx][0].toggle()       
            layoutCB.addWidget(self.data[cell_idx][0])
            layoutCB.setAlignment(QtCore.Qt.AlignCenter)
            layoutCB.setContentsMargins(0,0,0,0)
            
            self.tableWidget.setCellWidget(cell_idx, 0, cellWidget)
            self.tableWidget.setItem(cell_idx, 1, QtWidgets.QTableWidgetItem('{}'.format(self.data[cell_idx][1])))
            item = self.tableWidget.item(cell_idx, 1)
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.tableWidget.setItem(cell_idx, 2,  QtWidgets.QTableWidgetItem(self.data[cell_idx][2]))
            

        self.tableWidget2 = QtWidgets.QTableWidget()
        self.tableWidget2.setColumnCount(3)
        self.tableWidget2.setObjectName("tableWidget2")
        header2 = self.tableWidget2.horizontalHeader()
        self.tableWidget2.setColumnWidth(0,40)
        self.tableWidget2.setColumnWidth(1,40)
        header2.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        
        self.tableWidget.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.tableWidget.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)#cell 클릭 시 row 전체 선택 되도록
        self.tableWidget.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
    
        self.tableWidget2.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.tableWidget2.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tableWidget2.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        
    def set_table_contents(self):
        for cell_idx in range(len(self.data)):
            cellWidget = QtWidgets.QWidget()
            layoutCB = QtWidgets.QHBoxLayout(cellWidget)  
            layoutCB.addWidget(self.data[cell_idx][0])
            layoutCB.setAlignment(QtCore.Qt.AlignCenter)
            layoutCB.setContentsMargins(0,0,0,0)
            
            self.tableWidget.setCellWidget(cell_idx, 0, cellWidget)
            self.tableWidget.setItem(cell_idx, 1, QtWidgets.QTableWidgetItem('{}'.format(self.data[cell_idx][1])))
            item = self.tableWidget.item(cell_idx, 1)
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.tableWidget.setItem(cell_idx, 2,  QtWidgets.QTableWidgetItem(self.data[cell_idx][2]))
        
        for cell_idx in range(len(self.tmp_data)):
            cellWidget = QtWidgets.QWidget()
            layoutCB = QtWidgets.QHBoxLayout(cellWidget)
            layoutCB.addWidget(self.tmp_data[cell_idx][0])
            layoutCB.setAlignment(QtCore.Qt.AlignCenter)
            layoutCB.setContentsMargins(0,0,0,0)
            
            self.tableWidget2.setCellWidget(cell_idx, 0, cellWidget)
            self.tableWidget2.setItem(cell_idx, 1, QtWidgets.QTableWidgetItem('{}'.format(self.tmp_data[cell_idx][1])))
            item = self.tableWidget2.item(cell_idx, 1)
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.tableWidget2.setItem(cell_idx, 2,  QtWidgets.QTableWidgetItem(self.tmp_data[cell_idx][2]))
    
        
    def refresh_tables(self):
        self.data = sorted(self.data, key=lambda x: x[1])
        self.tmp_data = sorted(self.tmp_data, key=lambda x: x[1])
        
        self.tableWidget.setRowCount(len(self.data))
        self.tableWidget2.setRowCount(len(self.tmp_data))
        
        self.set_table_contents()
         
        
        
        
    def insert(self):
        for i in range(len(self.data)-1, -1, -1):
            if self.data[i][0].isChecked():
                self.data[i][0].setChecked(False)
                self.tmp_data.append(self.data[i])
                del(self.data[i])
            else:
                continue
        
        self.refresh_tables()
        
    def remove(self):
        for i in range(len(self.tmp_data)-1, -1, -1):
            if self.tmp_data[i][0].isChecked():
                self.data.append(self.tmp_data[i])
                del(self.tmp_data[i])
            else:
                continue
        
        self.refresh_tables()
    
    def left_select_all(self):
        for d in self.data:
            d[0].setChecked(True)
                
    def left_deselect_all(self):
        for d in self.data:
            d[0].setChecked(False)
            
    def right_select_all(self):
        for d in self.tmp_data:
            d[0].setChecked(True)
                
    def right_deselect_all(self):
        for d in self.tmp_data:
            d[0].setChecked(False)
            
    def get_idx(self):
        #checked_idx = []
        #for i in range(len(self.checkbox_list)):
        #    cbox = self.checkbox_list[i]
        #    if cbox.isChecked():
        #        checked_idx.append(i)
        #
        #return checked_idx
        checked_idx = [d[1] for d in self.tmp_data]
        return checked_idx
        
    
    def cancel(self):
        self.close()
    #-------------------------------
    
if __name__ == '__main__':
    None

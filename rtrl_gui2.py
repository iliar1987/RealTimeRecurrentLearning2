# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 13:09:21 2015

@author: Ilia
"""

from PyQt4 import QtGui,QtCore
import numpy as np
import colorsys
import cPickle
import matplotlib as mpl

import sys

try:    
    mpl.use(u'Qt4Agg',force=True)
except:
    print >> sys.stderr, "error loading mpl (no plotting will be available)"

from matplotlib import pyplot as plt

import RTRL_plot


#import os
#if os.name == 'nt':
#    iswindows = True    
#    import win32clipboard as clipboard
#else:
#    iswindows = False

def GetCurrentNx():
    return rnn1.nx

#class ParameterTable(QtGui.QTableView):
#    def __init__(self,parent):
#        super(ParameterTable,self).__init__(parent)
#        self.parent = parent
#        self.Populate()
#        self.AdjustGeometry()
#    def Populate(self):
#        model = QtGui.QStandardItemModel(4,2,self)
#        model.setHorizontalHeaderItem(0, QtGui.QStandardItem("Parameter"))
#        model.setHorizontalHeaderItem(1, QtGui.QStandardItem("Value"))
#        firstRow = QtGui.QStandardItem("ColumnValue")
#        model.setItem(0,0,firstRow)
#        self.setModel(model)
#        self.model = model
#    def AdjustGeometry():
#        rect = self.geometry()
#        int tableWidth = 2 + ui->tableWidget->verticalHeader()->width();
#        for(int i = 0; i < ui->tableWidget->columnCount(); i++){
#            tableWidth += ui->tableWidget->columnWidth(i);
#        }
#        tableHeight = 2 + ui->tableWidget->horizontalHeader()->height();
#        for(int i = 0; i < ui->tableWidget->rowCount(); i++){
#            tableHeight += ui->tableWidget->rowHeight(i);
#}        rect.setHeight(tableHeight);

class ParameterEditItem(QtGui.QLineEdit):
    def __init__(self,parent,main_widget,name,callback,index,value,validator,dtype):
        super(ParameterEditItem,self).__init__(value,parent)
        self.name = name
        self.index = index
        self.main_widget = main_widget
        self.parent = parent
        self.value = str(value)
        self.validator = validator
        self.setValidator(validator)
        self.callback = callback
        self.dtype=dtype
#        self.learning_rate_edit.returnPressed.connect(self.main_widget.setFocus)
#        editingFinished
        self.editingFinished.connect(self._OnChangeEvent)
        self.sending_signal = True
    def isSending(self):
        return self.sending_signal
    def StopSending(self):
        self.sending_signal=False
    def StartSending(self):
        self.sending_signal=True
    def _OnChangeEvent(self):
        if ( self.isSending() ):
            self.StopSending()
            self.callback(self)
            self.StartSending()
    def GetValue(self):
        return self.dtype(self.text())
    def SetValue(self,value):
        was_sending = self.isSending()
        if ( was_sending):
            self.StopSending()
        self.setText(str(value))
        if ( was_sending):
            self.StartSending()
        
from collections import defaultdict

class ParameterTable(QtGui.QTableWidget):
    def __init__(self,parent):
        super(ParameterTable,self).__init__(parent)
        self.parent = parent
        self.param_names = ['learning_rate','alpha','beta','nx']
        self._CreateEditControls()
        self._Populate()
        self.UpdateTableFromParams(params)
        self.AdjustGeometry()
    def _CreateEditControls(self):
        self.param_edits = {}
        
        param_change_events = defaultdict(lambda : self.ParamValueChanged)
        param_change_events['nx'] = self.NxChanged
        param_validators = {'learning_rate' : QtGui.QDoubleValidator(0.0,1.0,6),
                     'alpha' : QtGui.QDoubleValidator(-1.0,1.0,6),
                    'beta' : QtGui.QDoubleValidator(-1.0,1.0,6),
                    'nx':QtGui.QIntValidator(2,20)}
        param_dtypes = defaultdict(lambda : float)
        param_dtypes['nx'] = int
        for i,pname in enumerate(self.param_names):
            param_edit_item = ParameterEditItem(parent=self,
                                    main_widget = self.parent,
                                    name = pname,
                                    callback = param_change_events[pname],
                                    index = i,
                                    value = '',
                                    validator = param_validators[pname],
                                    dtype=param_dtypes[pname])
            self.param_edits[pname] = param_edit_item
    def _Populate(self):
        self.setRowCount(len(self.param_names))
        self.setColumnCount(2)
        self.setHorizontalHeaderLabels(["Parameter","Value"])
        for i,pname in enumerate(self.param_names):
            item_pname = QtGui.QTableWidgetItem(pname)
            item_pname.setFlags(\
                QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            self.setItem(i,0,item_pname)
            param_edit_item = self.param_edits[pname]
            self.setCellWidget(i,1,param_edit_item)
    def AdjustGeometry(self):
        tableWidth = 2+self.verticalHeader().width()
        for i in range (self.columnCount() ):
            tableWidth += self.columnWidth(i)
        tableHeight = 2+self.horizontalHeader().height()
        for i in range( self.rowCount() ):
            tableHeight += self.rowHeight(i)
        self.setFixedSize(tableWidth,tableHeight)
    def UpdateTableFromParams(self,params):
        for pname in self.param_names:
            if pname in params:
#                self.param_edits[pname].setText(params[pname])
                self.param_edits[pname].SetValue(params[pname])
            elif pname == 'nx':
                self.param_edits[pname].SetValue(GetCurrentNx())
    def UpdateParamsFromTable(self,params):
        for pname in self.param_names:
            if pname in params:
                params[pname] = self.param_edits[pname].GetValue()
    def NxChanged(self,item):
        if item.GetValue() != GetCurrentNx():
            result = QtGui.QMessageBox.question(self.parent,'Confirm Reset',
                                       'Changing this parameter requires reset, \ndo you want to reset the neural network?',
                                       QtGui.QMessageBox.Yes | QtGui.QMessageBox.No,QtGui.QMessageBox.No)
            if result == QtGui.QMessageBox.Yes:
                new_nx = self.param_edits['nx'].GetValue()
                self.parent.ResetNeuralNetwork(new_nx)
            else:
                self.param_edits['nx'].SetValue(GetCurrentNx())
        self.parent.setFocus()
    def ParamValueChanged(self,item):
        global params
        params[item.name] = item.GetValue()
        self.parent.setFocus()

    def DisableDeviationCosts(self):
        for pname in ['alpha','beta']:
            self.param_edits[pname].setEnabled(False)
    def EnableDeviationCosts(self):
        for pname in ['alpha','beta']:
            self.param_edits[pname].setEnabled(True)

class Plotter_Handler(object):
    def __init__(self,parent,rnn,plotter_type,my_checkbox=None):
        self.parent = parent
        self.rnn=rnn
        self.plotter_type = plotter_type
        self.weight_plotter = None
        my_checkbox.setCheckState(False)
        if not my_checkbox is None:
            my_checkbox.stateChanged.connect(self.OnClick_DisplayWeightPlot)
        self.my_checkbox = my_checkbox
        self.not_updating = False
    def StartWeightPlotter(self):
        RP = self.plotter_type()
        RP.RetrieveState(self.rnn)
#        print '1'
        RP.CreateAxes()
#        print '2'
        RP.CreateImages()
#        print '3'
        self.weight_plotter = RP
        self.weight_plotter.ConnectCloseEvent(self.OnWeightPlotterFigClose)
        fig = self.weight_plotter.GetFig()
        fig.canvas.mpl_connect('key_press_event', self._FigKeyPressed)
        plt.show(block=False)
        plt.ion()
    def _FigKeyPressed(self,evt):
        if ( evt.key == '0' or evt.key == '1'):
            self.parent.Insert_0_1(evt.key)
    
    def DestroyWeightPlotter(self):
        if not self.weight_plotter is None:
            self.weight_plotter.CloseFig()
            del(self.weight_plotter)
            self.weight_plotter=None
    
    def RestartWeightPlotter(self):
        if not self.weight_plotter is None:
            self.DestroyWeightPlotter()
            self.StartWeightPlotter()
    
    def UpdateWeightPlotter(self):
        if self.not_updating:
            return
        if not self.weight_plotter is None and self.weight_plotter.isActive():
            self.weight_plotter.RetrieveState(rnn1)
            self.weight_plotter.UpdateImages()
    def Freeze(self):
        self.not_updating = True
    def UnFreeze(self):
        self.not_updating = False
    def OnWeightPlotterFigClose(self,evt):
        self.my_checkbox.setCheckState(False)
        self.DestroyWeightPlotter()
    def OnClick_DisplayWeightPlot(self):
        if ( self.my_checkbox.checkState() ):
            self.StartWeightPlotter()
        else:
            self.DestroyWeightPlotter()

def InsertTextIntoEdit(edit,s):
    edit.moveCursor(QtGui.QTextCursor.End)
    edit.insertPlainText (s)
    edit.moveCursor(QtGui.QTextCursor.End)
def DeleteLastChar(edit):
    edit.moveCursor(QtGui.QTextCursor.End)
    edit.textCursor().deletePreviousChar()
    edit.moveCursor(QtGui.QTextCursor.End)

class MainWidget(QtGui.QMainWindow):
    
    def __init__(self):
        super(MainWidget, self).__init__()
        
        self.initUI()
        
    def initUI(self):
        
        QtGui.QToolTip.setFont(QtGui.QFont('SansSerif', 10))
        
        self.setToolTip('type 1\'s and 0\'s')
        
        exitAction = QtGui.QAction(QtGui.QIcon('exit.png'), '&Exit', self)        
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(self.close)
        
        saveStateAction = QtGui.QAction(QtGui.QIcon('icons/16x16/actions/document-save.png'), '&Save State As...', self)        
        saveStateAction.setShortcut('Ctrl+S')
        saveStateAction.setStatusTip('Save State As...')
        saveStateAction.triggered.connect(self.SaveStateAsAction)
        
        loadStateAction = QtGui.QAction(QtGui.QIcon('icons/16x16/actions/document-open.png'), '&Load State...', self)        
        loadStateAction.setShortcut('Ctrl+O')
        loadStateAction.setStatusTip('Load State...')
        loadStateAction.triggered.connect(self.LoadStateAction)
        
        insertSeqAction = QtGui.QAction(QtGui.QIcon('icons/16x16/actions/document-open.png'), '&Insert From File...',self)
        insertSeqAction.setStatusTip('Insert an input sequence from file...')
        insertSeqAction.triggered.connect(self.InsertSeqFromFileAction)
        
        saveInputAction = QtGui.QAction(QtGui.QIcon('icons/16x16/actions/document-save.png'), '&Save Inputs...',self)
        saveInputAction.setStatusTip('Save the inputs to a file...')
        saveInputAction.triggered.connect(self.SaveInputAction)
        
        saveGuessesAction = QtGui.QAction(QtGui.QIcon('icons/16x16/actions/document-save.png'), 'Save &Guesses...',self)
        saveGuessesAction.setStatusTip('Save the guesses to a file...')
        saveGuessesAction.triggered.connect(self.SaveGuessesAction)
        
        if True:
            pasteInputAction = QtGui.QAction(QtGui.QIcon('icons/16x16/actions/edit-paste.png'), '&Paste Input',self)
            pasteInputAction.setShortcut('Ctrl+V')
            pasteInputAction.setStatusTip('Paste Input Sequence from Clipboard...')
            pasteInputAction.triggered.connect(self.PasteInputFromClipboard)
            
            copyInputAction = QtGui.QAction(QtGui.QIcon('icons/16x16/actions/edit-copy.png'), '&Copy Input',self)
            copyInputAction.setShortcut('Ctrl+C')
            copyInputAction.setStatusTip('Copy Input Sequence to Clipboard...')
            copyInputAction.triggered.connect(self.CopyInputToClipboard)
        
        clearInputAction = QtGui.QAction(QtGui.QIcon('icons/16x16/actions/edit-clear-all.png'), 'C&lear',self)
        clearInputAction.setStatusTip('Copy Input and Guess Sequence...')
        clearInputAction.triggered.connect(self.ClearSequence)
        
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(saveStateAction)
        fileMenu.addAction(loadStateAction)
        fileMenu.addSeparator()
        fileMenu.addAction(exitAction)
        
        sequenceMenu = menubar.addMenu('&Sequence')
        sequenceMenu.addAction(insertSeqAction)
        sequenceMenu.addSeparator()
        sequenceMenu.addAction(saveInputAction)
        sequenceMenu.addAction(saveGuessesAction)
        #        if iswindows:
        if True:
            sequenceMenu.addSeparator()
            sequenceMenu.addAction(copyInputAction)
            sequenceMenu.addAction(pasteInputAction)
        sequenceMenu.addSeparator()
        sequenceMenu.addAction(clearInputAction)
        
        self.parameter_table = ParameterTable(self)
        self.deviationCostsCheckBox = QtGui.QCheckBox('Enable deviation costs (experimental): ',self)
        self.deviationCostsCheckBox.stateChanged.connect(self.deviationCostsCheckBoxTicked)

        if not rtrl2.UsingDeviationCosts():
            self.parameter_table.DisableDeviationCosts()
            self.deviationCostsCheckBox.setCheckState(False)
        else:
            self.parameter_table.EnableDeviationCosts()
            self.deviationCostsCheckBox.setCheckState(True)
#        self.clicked.connect(self.setFocus)
        
        self.draw_specgram_button = QtGui.QPushButton("Draw Spectrogram")
        self.draw_specgram_button.clicked.connect(self.DrawSpecGram)
        self.edit1 = QtGui.QTextEdit(self)
        self.edit1.setReadOnly(True)
        
        self.edit2 = QtGui.QTextEdit(self)
        self.edit2.setReadOnly(True)
        
        self.reset_count_button = QtGui.QPushButton('Reset Count',self)
        self.reset_count_button.clicked.connect(self.OnClick_ResetCount)
        
        self.reset_neural_network_button = QtGui.QPushButton('Reset\nNeural Network',self)
        self.reset_neural_network_button.clicked.connect(self.OnClick_reset_neural_network)
        
#        self.learning_rate_edit = QtGui.QLineEdit('%.3f'%params['learning_rate'],self)
#        self.learning_rate_edit.setValidator(QtGui.QDoubleValidator(0.0,1.0,6))
#        self.learning_rate_edit.returnPressed.connect(self.setFocus)
#        self.learning_rate_edit.editingFinished.connect(self.UpdateLearningRate)
#        self.learning_rate_hbox = QtGui.QHBoxLayout()
#        self.learning_rate_hbox.addWidget(QtGui.QLabel("learning rate = "))
#        self.learning_rate_hbox.addWidget(self.learning_rate_edit)
#        
        self.plotters = {}
        self.display_weight_plot_button = QtGui.QCheckBox('Display Weight Plot',self)
        self.plotters['weights'] = Plotter_Handler(self,rnn1,RTRL_plot.RTRL_Weight_Plotter,self.display_weight_plot_button)
        self.display_gradients_plot_button = QtGui.QCheckBox('Display Gradients Plot',self)
        self.plotters['gradients'] = Plotter_Handler(self,rnn1,RTRL_plot.RTRL_Gradients_Plotter,self.display_gradients_plot_button)
        
        self.sureness_label = QtGui.QLabel()
        self.sureness_color_label = QtGui.QLabel()
        self.sureness_color_pixmap = QtGui.QPixmap (50,25)
        self.sureness_color_pixmap.fill(QtGui.QColor("transparent"))
        self.sureness_color_label.setPixmap(self.sureness_color_pixmap)
        self.next_guess_sureness = 0.5
        self.UpdateSureness()
        
#        self.nx_label = QtGui.QLabel()
#        self.UpdateNxLabel()
        self.percent_label = QtGui.QLabel('',self)
        
        self.vbox1 = QtGui.QVBoxLayout()
        
        self.vbox1.addWidget(self.edit1,alignment = QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop )
        self.vbox1.addSpacing(10)
        self.vbox1.addWidget(self.edit2,alignment = QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.vbox1.addSpacing(10)
        self.vbox1.addWidget(self.sureness_label,alignment = QtCore.Qt.AlignCenter)
        self.vbox1.addSpacing(10)
        self.vbox1.addWidget(self.sureness_color_label,alignment = QtCore.Qt.AlignCenter)
        
        
        self.vbox2 = QtGui.QVBoxLayout()

        self.vbox2.addWidget(self.percent_label,alignment = QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.color_drawing_place_label = QtGui.QLabel()
        self.color_drawing_pixmap = QtGui.QPixmap (50,25)
        self.color_drawing_pixmap.fill(QtGui.QColor("transparent"))
        self.color_drawing_place_label.setPixmap(self.color_drawing_pixmap)
        self.vbox2.addSpacing(50)
        self.vbox2.addWidget(self.color_drawing_place_label,alignment = QtCore.Qt.AlignCenter)
        self.vbox2.addSpacing(50)
        self.vbox2.addWidget(self.reset_count_button,alignment = QtCore.Qt.AlignCenter)
        self.vbox2.addSpacing(10)
        self.vbox2.addWidget(self.reset_neural_network_button,alignment = QtCore.Qt.AlignCenter)
        self.vbox2.addSpacing(10)
#        self.vbox2.addLayout(self.learning_rate_hbox)
#        self.vbox2.addWidget(self.nx_label)
        self.vbox2.addSpacing(10)
        self.vbox2.addWidget(self.display_weight_plot_button)
        self.vbox2.addSpacing(10)
        self.vbox2.addWidget(self.display_gradients_plot_button)
        self.vbox2.addSpacing(10)
        self.vbox2.addWidget(self.draw_specgram_button,alignment = QtCore.Qt.AlignCenter)
        self.vbox2.addSpacing(20)
        self.vbox2.addWidget(self.deviationCostsCheckBox)
               
        
        self.hbox1 = QtGui.QHBoxLayout()
        dummy1=QtGui.QWidget()
        dummy1.setLayout(self.vbox1)
        self.hbox1.addWidget(dummy1,alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        
        
        dummy2=QtGui.QWidget()
        dummy2.setLayout(self.vbox2)
        self.hbox1.addWidget(dummy2,alignment=QtCore.Qt.AlignRight | QtCore.Qt.AlignTop)
        
        self.vbox4 = QtGui.QVBoxLayout()
        self.vbox4.addWidget(self.parameter_table)
        self.vbox4.addSpacing(150)
        self.hbox1.addSpacing(20)
        self.hbox1.addLayout(self.vbox4)
        

#        self.hbox2 = QtGui.QHBoxLayout()
#        self.hbox2.addSpacing(100)
#        self.hbox2.addSpacing(100)
        
        self.vbox3 = QtGui.QVBoxLayout()
        self.vbox3.addLayout(self.hbox1)
        self.vbox3.addSpacing(100)
#        self.vbox3.addSpacing(10)
#        self.vbox3.addLayout(self.hbox2)
        
        self.central_widget = QtGui.QWidget(self)
        self.central_widget.setLayout(self.vbox3)
        self.setCentralWidget(self.central_widget)
        self.UpdateBarColor()
        self.UpdatePercentLabel()
        
        self.UpdateNextGuess()
        
        
        self.setGeometry(300, 300,750, 200)
        self.setWindowTitle('Guesser')
        self.show()
#        self.parameter_table.AdjustGeometry()
        self.setFocus()
    def DrawSpecGram(self):
        x1_s = str(self.edit1.toPlainText())
        x1 = np.array([float(c) for c in x1_s])
        x2_s = str(self.edit2.toPlainText())
        x2 = np.array([float(c) for c in x2_s[:-1]])
        nfft = 16
        if (len(x1_s)<nfft):
            QtGui.QMessageBox.critical(self,"Error","Not enough values in input, need at least %d" % nfft)
            return
        plt.figure()
        a1 = plt.subplot(2,1,1)
        plt.specgram(x1,NFFT=nfft,Fs=1,noverlap=nfft/2)
        plt.title('inputs')
        plt.ylabel('frequency')
        a2 = plt.subplot(2,1,2,sharex=a1,sharey=a1)
        plt.specgram(x2,NFFT=nfft,Fs=1,noverlap=nfft/2)
        plt.title('guesses')
        plt.ylabel('frequency')
        plt.xlabel('time')
#        plt.gcf().canvas.setParent(self)
        plt.show(block=False)
        plt.ion()
        
    def deviationCostsCheckBoxTicked(self,state):
        result = QtGui.QMessageBox.question(self,'Confirm Reset',
                                       'Changing this parameter requires reset, \ndo you want to reset the neural network?',
                                       QtGui.QMessageBox.Yes | QtGui.QMessageBox.No,QtGui.QMessageBox.No)
        if result == QtGui.QMessageBox.Yes:
            state_bool= state == QtCore.Qt.Checked
            if state_bool:
                rtrl2.EnableDeviationCosts()
            else:
                rtrl2.DisableDeviationCosts()
            params['alpha'] = RTRL2.alpha_default_value
            params['beta'] = RTRL2.beta_default_value
            if state_bool:
                self.parameter_table.EnableDeviationCosts()
            else:
                self.parameter_table.DisableDeviationCosts()
            self.HardResetNeuralNetwork(rnn1.nx)
    def InsertSequence(self,s):
        self.FreezePlotters()
        for c in s:
            if c!='0' and c!='1':
                raise Exception('bad character encountered in file: %c = %d' % (c,ord(c)))
            self.Insert_0_1(c)
        self.UnFreezePlotters()
    def ClearSequence(self):
        self.edit1.setText('')
        self.edit2.setText('*')
#    if iswindows:
    if True:
        def PasteInputFromClipboard(self):
            try:
                cb = QtGui.QApplication.clipboard()
                s=str(cb.text(mode=cb.Clipboard))
            except:
                print >> sys.stderr, 'couldn\'t access clipboard'
            else:
                self.InsertSequence(s)
        def CopyInputToClipboard(self):
            s = str(self.edit1.toPlainText())
            try:
                cb = QtGui.QApplication.clipboard()
                cb.clear(mode=cb.Clipboard )
                cb.setText(s,mode=cb.Clipboard)
            except:
                print >> sys.stderr, 'couldn\'t access clipboard'

    def InsertSeqFromFileAction(self):
        fname = QtGui.QFileDialog.getOpenFileName(\
            self,
            "Load Input Sequence from File:",
            ".",
            filter = ("Sequence Files (*.seq);;Text Files (*.txt);;All Files (*.*)"))
        if not len(fname)==0:
            with open(fname,'rb') as f:
                s = f.readline()
            self.InsertSequence(s)
    def SaveInputAction(self):
        fname = QtGui.QFileDialog.getSaveFileName(\
            self,
            "Save Input Sequence As:",
            ".",
            filter = ("Sequence Files (*.seq);;Text Files (*.txt);;All Files (*.*)"))
        if not len(fname)==0:
            with open(fname,'wb') as f:
                f.writelines([str(self.edit1.toPlainText())])
    def SaveGuessesAction(self):
        fname = QtGui.QFileDialog.getSaveFileName(\
            self,
            "Save Guesses Sequence As:",
            ".",
            filter = ("Sequence Files (*.seq);;Text Files (*.txt);;All Files (*.*)"))
        if not len(fname)==0:
            with open(fname,'wb') as f:
                f.writelines([str(self.edit2.toPlainText())[:-1]])
    def mousePressEvent(self,e):
        self.setFocus()
    def UpdateSureness(self):
        self.last_guess_sureness = self.next_guess_sureness
        self.UpdateSurenessLabel()
        self.UpdateSurenessColor()
    def UpdateSurenessLabel(self):
        self.sureness_label.setText('sureness of last guess = %.1f%%'%(self.last_guess_sureness*100.0))
    def UpdateSurenessColor(self):
        if len(ys)>0:
            last_guess_correct = ys[-1] == vs[-1]
            if last_guess_correct:
                coefficient = -1.0
            else:
                coefficient = 1.0
        else:
            coefficient = 0.0
        number = coefficient*(2.0*(self.last_guess_sureness-0.5))**2
        h_min = 0.0
        h_max = 1./3
        h_middle = 0.5*(h_max + h_min)
        h_amplitude = h_max-h_middle
        h_color = h_middle + number * h_amplitude
        
        s_color = 2*(self.last_guess_sureness-0.5)
        
        bar_color = np.array(colorsys.hsv_to_rgb(h_color, s_color, 0.9))
        c = np.asarray(bar_color * 255,dtype=int)
        qt_color = QtGui.QColor(*c)
        self.sureness_color_pixmap.fill(qt_color)
        self.sureness_color_label.setPixmap(self.sureness_color_pixmap)
#    def UpdateLearningRate(self):
#        global params
#        params['learning_rate'] = float(self.learning_rate_edit.text())
#    def UpdateNxLabel(self):
#        self.nx_label.setText('nx = %d'%rnn1.nx)
    def LoadStateAction(self):
        fname = QtGui.QFileDialog.getOpenFileName(\
            self,
            "Load Network State File:",
            ".",
            filter = ("Neural Network State Files (*.rtrl2);;All Files (*.*)"))
        if not len(fname)==0:
            with open(fname,'rb') as f:
                state = cPickle.load(f)
            rnn1.SetState(state)
            
#            if 'learning_rate' in state:
#                global params
#                params['learning_rate'] = state['learning_rate']
            UpdateParamsFromState(state)
            self.UpdateParamTable()
#            self.learning_rate_edit.setText('%.5f'%params['learning_rate'])
#            self.UpdateNxLabel()
            self.OnClick_ResetCount()
            self.RestartPlotters()
#        print fname
    def SaveStateAsAction(self):
        fname = QtGui.QFileDialog.getSaveFileName(\
            self,
            "Save Network State File As:",
            ".",
            filter = ("Neural Network State Files (*.rtrl2);;All Files (*.*)"))
        if not len(fname)==0:
            with open(fname,'wb') as f:
                state = rnn1.GetState()
#                state['learning_rate'] = params['learning_rate']
                state = UpdateStateFromParams(state)
                cPickle.dump(state,f)
#        print fname
    def OnClick_ResetCount(self):
        global successes,N_total
        N_total=0
        successes=0
        self.UpdateBarColor()
        self.UpdatePercentLabel()
    def GetInputNx(self):
        result = QtGui.QInputDialog.getInt(self,"Set nx",
                                  "nx = ",
                                  value=4,
                                  min=1,
                                  max=20)        
        return result
    def RestartPlotters(self):
        for v in self.plotters.itervalues():
            v.RestartWeightPlotter()
    def FreezePlotters(self):
        for v in self.plotters.itervalues():
            v.Freeze()
    def UnFreezePlotters(self):
        for v in self.plotters.itervalues():
            v.UnFreeze()
        self.UpdatePlotters()
    def UpdatePlotters(self):
        for v in self.plotters.itervalues():
            v.UpdateWeightPlotter()
    def UpdateParamTable(self):
        self.parameter_table.UpdateTableFromParams(params)
    def HardResetNeuralNetwork(self,new_nx):
        HardResetNetwork()
        self.ResetNeuralNetwork(new_nx)
    def ResetNeuralNetwork(self,new_nx):
        rnn1.Reset(new_nx)
        self.OnClick_ResetCount()
#            self.UpdateNxLabel()
        self.UpdateParamTable()
        self.RestartPlotters()
    def OnClick_reset_neural_network(self):
        get_nx_result = self.GetInputNx()
        if get_nx_result[1]:
            new_nx = get_nx_result[0]
            self.ResetNeuralNetwork(new_nx)
        
    def UnvielGuess(self):
        DeleteLastChar(self.edit2)
        InsertTextIntoEdit(self.edit2,'%d'%self.next_guess)
        
    def UpdateNextGuess(self):
        self.next_guess,self.next_guess_sureness = GetRTRL_Guess(return_sureness=True)
        InsertTextIntoEdit(self.edit2,'*')
        
    def UpdatePercentLabel(self):
        s = '%d / %d' % (successes,N_total)
        if N_total>0:
            s = s + ' = %.1f%%' % (100.0*float(successes)/N_total)
#        print s
        self.percent_label.setText(s)
    
    def UpdateBarColor(self):
        take_last = 15
        if N_total>0:
            if N_total>take_last:
                percentage = sum([ys[-i] == vs[-i] for i in range(1,take_last+1)])/float(take_last)
            else:
                percentage = float(successes)/N_total
            h_min = 0.0
            h_max = 1./3
            this_h = h_min + (1.0-percentage) * (h_max-h_min)
            self.bar_color = np.array(colorsys.hsv_to_rgb(this_h, 1.0, 0.9))
        else:
            self.bar_color = np.array([0.,0.,0.])
        c = np.asarray(self.bar_color * 255,dtype=int)
        qt_color = QtGui.QColor(*c)
        self.color_drawing_pixmap.fill(qt_color)
        self.color_drawing_place_label.setPixmap(self.color_drawing_pixmap)
    def Insert_0_1(self,c): #c is '0' or '1'
        InsertTextIntoEdit(self.edit1,c)
        MakeRTRL_Step(c)
        self.UnvielGuess()
        global N_total,successes
        N_total+=1
        vs.append(int(c))
        ys.append(self.next_guess)
        self.UpdateSureness()
        if ( self.next_guess == int(c) ):
            successes+=1
        self.UpdatePercentLabel()
        self.UpdateBarColor()
        self.UpdateNextGuess()
        self.UpdatePlotters()
    def keyPressEvent(self, e):
        k = e.key()
        c = e.text()
        if k == QtCore.Qt.Key_Escape:
            self.close()
        elif c == '0' or c == '1':
            self.Insert_0_1(c)


import rtrl2
from rtrl2 import RTRL2

default_params = {'learning_rate':0.2,
                  'alpha':RTRL2.alpha_default_value,
                  'beta':RTRL2.beta_default_value}

params = dict(**default_params)

def UpdateParamsFromState(state):    
    global params
    state_params = state['extra_params']
    for k in params.iterkeys():
        if k in state_params:            
            params[k] = state_params[k]
        else:
            params[k] = default_params[k]

def UpdateStateFromParams(state):
#    for k in params:
#        state[k] = params[k]
    state['extra_params'] = params
    return state

nx_default = 4
successes=0
N_total=0
ys = []
vs = []
starting_state_fname = 'nx8.rtrl2'

rnn1=[]
def HardResetNetwork():
    global rnn1
    rnn1 = RTRL2()
    rnn1.BuildNetwork()

HardResetNetwork()


if not starting_state_fname is None:
    try:
        with open(starting_state_fname,'rb') as f:
            state = cPickle.load(f)
        rnn1.SetState(state)
        UpdateParamsFromState(state)
    except:
        print >> sys.stderr, "could not open default network state file: %s, continuing with randomized network."%starting_state_fname
        rnn1.Reset(nx_default)

def GetRTRL_Guess(return_sureness=False):
    next_y = rnn1.Get_Y_Prediction()
    guess=np.int(np.round(next_y))
    ret_tuple = (guess,)
    calc_sureness = lambda y,guess: y if guess else (1.0-y)
    if return_sureness:
        ret_tuple = ret_tuple + (calc_sureness(next_y,guess),)
    return ret_tuple

def MakeRTRL_Step(v_char):
    v = np.array([int(v_char)])
    rnn1.MakeStep(v,**params)

def main():
    app = QtGui.QApplication([])
    ex = MainWidget()
    app.exec_()

if __name__ == '__main__':
    
    main()

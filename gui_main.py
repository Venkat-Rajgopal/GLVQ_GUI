###########################################################################
# GUI implementation for LVQ family algorithms (GLVQ, GMLVQ, GRLVQ, LGMLVQ)

# Author: Venkatramani Rajgopal
# April 2018
###########################################################################

# Matplotlib for plotting

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import style
from matplotlib import pyplot as plt
# plot style under matplotlib
style.use("fivethirtyeight")

# Inbuilt python tools for iteration 
import random
import itertools

# pandas and numpy for data processing. 
import pandas as pd
import numpy as np

# import sklearn library for datasets, preprocessing and pca
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.decomposition import PCA

# GUI is built on tkinter
import tkinter as tk
from tkinter import ttk
from tkinter import *
from tkinter.filedialog import askopenfilename


# import models from lvq library
from glvq import GlvqModel, GrlvqModel, LgmlvqModel, GmlvqModel, plot2d
from glvq.plot_2d import to_tango_colors, tango_color

# Different fonts in the window
LARGE_FONT= ("Verdana", 12)
NORM_FONT = ("Verdana", 10)
SMALL_FONT = ("Verdana", 8)

# saving some variables
creds_1 = 'inputsaver.csv'

# Text to display in the main window
explanation = """Scikit-learn compatible implementation of GLVQ, GRLVQ, GLMVQ and LGMLVQ. Python package Author Joris Jensen."""
explain2 = """Package home page: https://github.com/MrNuggelz/sklearn-glvq """
# ===================================================================================================
# Get first 2 principle components 
sklearn_pca = sklearnPCA(n_components=2)

""" In this section we define various functions which will be called from the title and menu bar page """

# collecting user clicks in a separate class
class inputscollector:
    # save the original data and its labels
    selected_data = []
    selected_labels = []

    # saving training and testing samples after splitting the dataset. 
    computed_Xtrain = []
    computed_ytrain = []
    computed_Xtest = []
    computed_ytest = []

    # save the prototypes user enters
    entered_prototypes = []

    # percentage of datasplit 
    getp = []

    # saving the predictions
    predictions = []

    # saving the coefficients
    glvq_coef_1 = []
    glvq_coef_2 = []
    glvq_colors = [] 

    # saving label names for confusion matrix plot
    selected_labelnames = []

    # save the sample size for cross validation
    gets = []

# function for pop up message. 
def popupmsg(msg):
    popup = tk.Tk()
    popup.wm_title("Message")

    # make a label on the window frame
    label = ttk.Label(popup, text=msg, font=NORM_FONT)
    label.pack(side="top", padx=10, pady=10)

    # place buttons
    B1 = ttk.Button(popup, text="Okay", command = popup.destroy)
    B1.pack(side="top")

    # window size 
    popup.geometry("300x200")
    popup.mainloop()


# confusion matrix function
def plot_confusion_matrix(cm, classes, mtd, title='Confusion matrix', cmap=plt.cm.Blues):

    fig1 = plt.figure(figsize=(4.5, 4.5))       # fig size

    # show the confusion matrix
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)

    # x and y axis ticks
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    
    # iterate through the matrix dimensions
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.grid()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def import_csv_data():
    #p = tk.Tk()
    #p.wm_title("Import data as CSV")    

    csv_file_path = askopenfilename()
    print(csv_file_path)
    data = pd.read_csv(csv_file_path)

    datas = {'data' : np.asarray(data.drop(['label_name','label'], axis=1)),
            'labels' : data['label'],
            'label_name' : np.unique(data['label_name'])}

    inputscollector.selected_data = datas['data']
    inputscollector.selected_labels = datas['labels']
    inputscollector.selected_labelnames = datas['label_name']

    # inputscollector.selected_data = data.drop(['label_name','label'], axis=1)
    # inputscollector.selected_labels = data.iloc[:,-1]
    # inputscollector.selected_labelnames = np.unique(data['label_name'])

    #return data

    #p.geometry("300x100")
    #p.mainloop() 



def before_train_msg(model_name):
    p1 = tk.Tk()
    p1.wm_title("Message")
    msg = ttk.Label(p1, text='Ensure data is split to training and test before training!', font=NORM_FONT).pack(side='top', padx=10, pady=10)
    #B1 = ttk.Button(p1, text="Okay", command = p1.destroy)
    B1 = ttk.Button(p1, text="Done splitting. Go ahead with training.", command=lambda: changeModel(model_name))
    B1.pack(side='top')
    #B2.pack(side='top')
    p1.geometry("400x100")
    p1.destroy
    #p1.mainloop() 


""" Main Model Function (Computes GLVQ, GMLVQ, grlvq, LGMLVQ) """


def changeModel(model_name):
    # displays this pop-up window on clicking the model_name
    popup = tk.Tk()
    popup.wm_title("Model Implementation")

    # function to make a label entry
    def makeentry(parent, caption, width=None, **options):
        # make label and pack to top
        ttk.Label(parent, text=caption, font=NORM_FONT).pack(side='top')
        # create entry and pack. 
        entry = ttk.Entry(parent, **options)
        if width:
            entry.config(width=width)
        entry.pack(side='top')
        return entry
    
    # conditional loop for the selected model. 
    # loop for GLVQ
    if model_name == 'GLVQ':
        # create header label
        label = ttk.Label(popup, text='Generalized learning vector quantization', font=NORM_FONT)
        label.pack(side="top", padx=10, pady=10)

        # entry widget to enter prototypes
        p_entry = makeentry(popup, "Enter Prototypes per class:", 10)
        # default prototypes as 3
        p_entry.insert(0,'1')
        p_entry.focus_set()

        # callback to get the chosen entry
        def callback():
            protos = (p_entry.get())
            # update the variable to the collector class
            inputscollector.entered_prototypes = protos

            # prt.config so that the text is printed in the GUI itself 
            prt.config(text='No. of prototypes selected:' + str(protos))   

            return protos        

        B1 = ttk.Button(popup, text='Save', command = callback)
        B1.pack(side="top", padx=10, pady=10)

        prt = Label(popup, text='No. of prototypes selected:', font=NORM_FONT)
        prt.pack(side='top', padx=10, pady=10)

        # glvq training functions
        def glvq_train():
            # call the model with the entered prototypes
            glvq = GlvqModel(prototypes_per_class=int(p_entry.get()))
            print(inputscollector.computed_Xtrain)
            print(inputscollector.computed_ytrain)

            # fit the model to the training data
            glvq.fit(inputscollector.computed_Xtrain, inputscollector.computed_ytrain)

            # compute model accuracies
            train_score = glvq.score(inputscollector.computed_Xtrain, inputscollector.computed_ytrain)
            test_score = glvq.score(inputscollector.computed_Xtest, inputscollector.computed_ytest)

            # print results to the GUI
            result.config(text='Results from GLVQ Classification:' + '\n' + '\n' + str('Training Accuracy = ') + str(train_score) + 
                            '\n' + str('Test Accuracy = ') + str(test_score))

            # predict on the test set
            pred = glvq.predict(inputscollector.computed_Xtest) 

            # update variables to the collector class
            inputscollector.predictions = pred
            inputscollector.glvq_coef_1 = glvq.w_[:, 0]
            inputscollector.glvq_coef_2 = glvq.w_[:, 1]
            inputscollector.glvq_colors = glvq.c_w_

        # button to train the model. 
        trainmodel = ttk.Button(popup, text='Train', command = glvq_train)
        trainmodel.pack(padx=10, pady=10, side='top')

        # folds window for cross validation
        def get_folds():
            # make a global variable
            global foldentry

            popfold = tk.Tk()
            popfold.wm_title("No of folds")
            flds = Label(popfold, text='Enter no of folds for Cross val:', font=NORM_FONT).pack(side='top', padx=10, pady=10)
            foldentry = makeentry(popfold, "No of folds:", 10)
            # default value
            foldentry.insert(0,'5')
            foldentry.focus_set()

            fldsbot = ttk.Button(popfold, text='Compute', command = cross_validate_glvq).pack(side='top', padx=10, pady=10)
            popfold.destroy

        # cross validation function    
        def cross_validate_glvq():

            glvq = GlvqModel(prototypes_per_class=int(p_entry.get()))
            clf  = glvq.fit(inputscollector.computed_Xtrain, inputscollector.computed_ytrain)

            # compute cross val with scikit learn cross_val_score
            scores = cross_val_score(clf, inputscollector.computed_Xtest, inputscollector.computed_ytest, cv=int(foldentry.get()))
            print("Mean Accuracy (0.95 CI) :  %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
            # compute the average score
            mean_score = np.mean(scores).astype(str)
            # print to the GUI
            cvscores.config(text='--------------------------------\n' + 'Cross Validations:' + '\n' + '\n' + str('Average Cross Validation score = ') + str(mean_score)) 

        crossval = ttk.Button(popup, text='Cross Validate', command = get_folds)
        crossval.pack(padx=10, pady=10)

        # plot the first two features
        def glvq_plot():
            pred = inputscollector.predictions
            plt.scatter(inputscollector.computed_Xtrain[:, 0], inputscollector.computed_Xtrain[:, 1], c=to_tango_colors(inputscollector.computed_ytrain), alpha=0.5)
            plt.scatter(inputscollector.computed_Xtrain[:, 0], inputscollector.computed_Xtrain[:, 1], c=to_tango_colors(pred), marker='.')
            plt.scatter(inputscollector.glvq_coef_1, inputscollector.glvq_coef_2,c=tango_color('aluminium', 5), marker='D')
            plt.scatter(inputscollector.glvq_coef_1,inputscollector.glvq_coef_2,c=to_tango_colors(inputscollector.glvq_colors, 0), marker='.')
            #plt.axis('equal')
            plt.grid()
            plt.show() 

        r = ttk.Button(popup, text='Show plots', command = glvq_plot)
        r.pack(padx=10, pady=10, side='bottom')

        # PCA computation
        def glvq_pca():

            # Fit the model with input data and apply the dimensionality reduction on it.
        	Y_sklearn = sklearn_pca.fit_transform(inputscollector.selected_data)
        	fig, axes = plt.subplots(ncols=2, figsize=(10, 4))

            # use axes to make 2 subplots. 
        	ax = axes[0]
        	ax.scatter(Y_sklearn[:, 0], Y_sklearn[:, 1], c=inputscollector.selected_labels, alpha=0.5)
        	ax.set_xlabel('Component 1')
        	ax.set_ylabel('Component 2')
        	ax.set_title("Component 1 vs 2")
        	ax.grid()

            # fit the model with selected data
        	pca = PCA().fit(inputscollector.selected_data)
        	ax = axes[1]
        	ax.plot(np.cumsum(pca.explained_variance_ratio_))
        	ax.set_xlabel('Number of components')
        	ax.set_ylabel('Cumulative Variance')
        	ax.set_title('Cumulative Variance explained')
        	plt.grid()
        	plt.show()

        pcaplot = ttk.Button(popup, text='Show PCA', command = glvq_pca)
        pcaplot.pack(padx=10, pady=10, side='bottom')

        # function to make the confusion matrix and plot based on the previous defined plot_confusion_matrix(). 
        def confmat():
            cm = confusion_matrix(inputscollector.computed_ytest, inputscollector.predictions)
            plot_confusion_matrix(cm, classes = inputscollector.selected_labelnames, mtd='FC_', title='Confusion matrix_GLVQ')

        # buttons for confusion matrix
        cmbutton = ttk.Button(popup, text='Show Confusion Matrix', command = confmat)
        cmbutton.pack(padx=10, pady=10, side='bottom')

        # label results section
        result = Label(popup, text='Results from GLVQ Classification:', font=NORM_FONT)
        result.pack(side='top', padx=10, pady=10)

        cvscores = Label(popup, text='--------------------------------\n' + 'Cross Validations:', font=NORM_FONT)
        cvscores.pack(side='top', padx=10, pady=10)

        # validate the results to be printed on the GUI window itself. 
        # without this gui won't print on the window. 
        def validate():
            result.config(state=(NORMAL if P else DISABLED))
            prt.config(state=(NORMAL if P else DISABLED))
            cvscores.config(state=(NORMAL if P else DISABLED))

            return True

           
    # LOOPING FOR GMLVQ (Most functions are the same as the previous model.)
    # Here the model imports gmlvq  

    if model_name == 'GMLVQ':
        label = ttk.Label(popup, text='Generalized Matrix Learning Vector Quantization', font=NORM_FONT)
        label.pack(side="top", padx=10, pady=10)

        p_entry = makeentry(popup, "Enter Prototypes per class:", 10)
        p_entry.insert(0,'1')
        p_entry.focus_set()

        def callback_1():
            protos = (p_entry.get())
            inputscollector.entered_prototypes = protos
            print("No. of prototypes selected:", protos)

            prt.config(text='No. of prototypes selected:' + str(protos))   
            return protos

        B1 = ttk.Button(popup, text='Save', command = callback_1)
        B1.pack(side="top", padx=10, pady=10)

        prt = Label(popup, text=':', font=NORM_FONT)
        prt.pack(side='top', padx=10, pady=10)

        def gmlvq_train():
            gmlvq = GmlvqModel(prototypes_per_class=int(p_entry.get()))
            gmlvq.fit(inputscollector.computed_Xtrain, inputscollector.computed_ytrain)
            train_score = gmlvq.score(inputscollector.computed_Xtrain, inputscollector.computed_ytrain)
            test_score = gmlvq.score(inputscollector.computed_Xtest, inputscollector.computed_ytest)

            result.config(text='Results from GMLVQ Classification:' + '\n' + '\n' + str('Training Accuracy=') + str(train_score) + 
                            '\n' + str('Test Accuracy=') + str(test_score))            
            
            pred = gmlvq.predict(inputscollector.computed_Xtest) 
            inputscollector.predictions = pred


        trainmodel = ttk.Button(popup, text='Train', command = gmlvq_train)
        trainmodel.pack(padx=10, pady=10, side='top')


        def get_folds():
            global foldentry_1

            popfold = tk.Tk()
            popfold.wm_title("No of folds")
            flds = Label(popfold, text='Enter no of folds for Cross val:', font=NORM_FONT).pack(side='top', padx=10, pady=10)
            foldentry_1 = makeentry(popfold, "No of folds:", 10)
            foldentry_1.insert(0,'5')
            foldentry_1.focus_set()

            fldsbot = ttk.Button(popfold, text='Compute', command = cross_validate_gmlvq).pack(side='top', padx=10, pady=10)
            popfold.destroy

        def cross_validate_gmlvq():

            gmlvq = GmlvqModel(prototypes_per_class=int(p_entry.get()))
            clf  = gmlvq.fit(inputscollector.computed_Xtrain, inputscollector.computed_ytrain)

            scores = cross_val_score(clf, inputscollector.computed_Xtest, inputscollector.computed_ytest, cv=int(foldentry_1.get()))
            mean_score = np.mean(scores).astype(str)
            cvscores.config(text='--------------------------------\n' + 'Cross Validations:' + '\n' + '\n' + str('Average Cross Validation score = ') + str(mean_score)) 

        crossval = ttk.Button(popup, text='Cross Validate', command = get_folds)
        crossval.pack(padx=10, pady=10)        

        def gmlvq_plot():
            gmlvq = GmlvqModel(prototypes_per_class=int(p_entry.get()))
            gmlvq.fit(inputscollector.computed_Xtrain, inputscollector.computed_ytrain)            
            plot2d(gmlvq, inputscollector.computed_Xtest, inputscollector.computed_ytest, 1, 'GMLVQ')
            plt.grid()
            plt.show()         

        r = ttk.Button(popup, text='Show plots', command = gmlvq_plot)
        r.pack(padx=10, pady=10, side='bottom')

        def confmat():
            cm = confusion_matrix(inputscollector.computed_ytest, inputscollector.predictions)
            plot_confusion_matrix(cm, classes = inputscollector.selected_labelnames, mtd='FC_', title='Confusion matrix_GMLVQ')

        cmbutton = ttk.Button(popup, text='Show Confusion Matrix', command = confmat)
        cmbutton.pack(padx=10, pady=10, side='bottom')

        result = Label(popup, text='Results from GMLVQ Classification:', font=NORM_FONT)
        result.pack(side='top', padx=10, pady=10)

        cvscores = Label(popup, text='--------------------------------\n' + 'Cross Validations:', font=NORM_FONT)
        cvscores.pack(side='top', padx=10, pady=10)        

        def validate():
            result.config(state=(NORMAL if P else DISABLED))
            prt.config(state=(NORMAL if P else DISABLED))
            cvscores.config(state=(NORMAL if P else DISABLED))
            return True

    # LOOPING FOR GRLVQ

    if model_name == 'GRLVQ':
        label = ttk.Label(popup, text='Generalized Relevance Learning Vector Quantization', font=NORM_FONT)
        label.pack(side="top", padx=10, pady=10)

        p_entry = makeentry(popup, "Enter Prototypes per class:", 10)
        p_entry.insert(0,'1')
        p_entry.focus_set()

        def callback_1():
            protos = (p_entry.get())
            inputscollector.entered_prototypes = protos
            print("No. of prototypes selected:", protos)

            prt.config(text='No. of prototypes selected:' + str(protos))   
            return protos

        B1 = ttk.Button(popup, text='Save', command = callback_1)
        B1.pack(side="top", padx=10, pady=10)
        #B2.pack()

        prt = Label(popup, text=':', font=NORM_FONT)
        prt.pack(side='top', padx=10, pady=10)

        def grlvq_train():
            grlvq = GrlvqModel(prototypes_per_class=int(p_entry.get()))
            grlvq.fit(inputscollector.computed_Xtrain, inputscollector.computed_ytrain)
            train_score = grlvq.score(inputscollector.computed_Xtrain, inputscollector.computed_ytrain)
            test_score = grlvq.score(inputscollector.computed_Xtest, inputscollector.computed_ytest)

            result.config(text='Results from GRLVQ Classification:' + '\n' + '\n' + str('Training Accuracy=') + str(train_score) + 
                            '\n' + str('Test Accuracy=') + str(test_score))            
            
            pred = grlvq.predict(inputscollector.computed_Xtest) 
            inputscollector.predictions = pred

        trainmodel = ttk.Button(popup, text='Train', command = grlvq_train)
        trainmodel.pack(padx=10, pady=10, side='top')

        def get_folds():
            global foldentry_2

            popfold = tk.Tk()
            popfold.wm_title("No of folds")
            flds = Label(popfold, text='Enter no of folds for Cross val:', font=NORM_FONT).pack(side='top', padx=10, pady=10)
            foldentry_2 = makeentry(popfold, "No of folds:", 10)
            foldentry_2.insert(0,'5')
            foldentry_2.focus_set()

            fldsbot = ttk.Button(popfold, text='Compute', command = cross_validate_grlvq).pack(side='top', padx=10, pady=10)
            popfold.destroy

        def cross_validate_grlvq():

            grlvq = GrlvqModel(prototypes_per_class=int(p_entry.get()))
            clf  = grlvq.fit(inputscollector.computed_Xtrain, inputscollector.computed_ytrain)

            scores = cross_val_score(clf, inputscollector.computed_Xtest, inputscollector.computed_ytest, cv=int(foldentry_2.get()))
            mean_score = np.mean(scores).astype(str)
            cvscores.config(text='--------------------------------\n' + 'Cross Validations:' + '\n' + '\n' + str('Average Cross Validation score = ') + str(mean_score)) 

        crossval = ttk.Button(popup, text='Cross Validate', command = get_folds)
        crossval.pack(padx=10, pady=10)                

        def grlvq_plot():
            grlvq = GrlvqModel(prototypes_per_class=int(p_entry.get()))
            grlvq.fit(inputscollector.computed_Xtrain, inputscollector.computed_ytrain)            
            plot2d(grlvq, inputscollector.computed_Xtest, inputscollector.computed_ytest, 1, 'GRLVQ')
            plt.grid()
            plt.show()         

        r = ttk.Button(popup, text='Show plots', command = grlvq_plot)
        r.pack(padx=10, pady=10, side='bottom')

        def confmat():
            cm = confusion_matrix(inputscollector.computed_ytest, inputscollector.predictions)
            plot_confusion_matrix(cm, classes = inputscollector.selected_labelnames, mtd='FC_', title='Confusion matrix_GRLVQ')

        cmbutton = ttk.Button(popup, text='Show Confusion Matrix', command = confmat)
        cmbutton.pack(padx=10, pady=10, side='bottom')

        result = Label(popup, text='Results from GRLVQ Classification:', font=NORM_FONT)
        result.pack(side='top', padx=10, pady=10)

        cvscores = Label(popup, text='--------------------------------\n' + 'Cross Validations:', font=NORM_FONT)
        cvscores.pack(side='top', padx=10, pady=10)                

        def validate():
            result.config(state=(NORMAL if P else DISABLED))
            prt.config(state=(NORMAL if P else DISABLED))
            cvscores.config(state=(NORMAL if P else DISABLED))
            return True

    # LOOPING FOR LGMLVQ

    if model_name == 'LGMLVQ':
        label = ttk.Label(popup, text='Localized Generalized Matrix Learning Vector Quantization', font=NORM_FONT)
        label.pack(side="top", padx=10, pady=10)

        p_entry = makeentry(popup, "Enter Prototypes per class:", 10)
        p_entry.insert(0,'1')
        p_entry.focus_set()

        def callback_3():
        	if int(p_entry.get()) >= 4:
        		label = ttk.Label(popup, font=NORM_FONT, command = popupmsg('Too many prototypes chosen. \nComputation takes longer. \nLesser selection is advised.')).pack()

        	else:
        		protos = (p_entry.get())
        		inputscollector.entered_prototypes = protos
        		print("No. of prototypes selected:", protos)
        		prt.config(text='No. of prototypes selected:' + str(protos))

        	#return protos

        B1 = ttk.Button(popup, text='Save', command = callback_3)
        B1.pack(side="top", padx=10, pady=10)
        #B2.pack()

        prt = Label(popup, text=':', font=NORM_FONT)
        prt.pack(side='top', padx=10, pady=10)

        def lgmlvq_train():
            lgmlvq = LgmlvqModel(prototypes_per_class=int(p_entry.get()))
            lgmlvq.fit(inputscollector.computed_Xtrain, inputscollector.computed_ytrain)
            train_score = lgmlvq.score(inputscollector.computed_Xtrain, inputscollector.computed_ytrain)
            test_score = lgmlvq.score(inputscollector.computed_Xtest, inputscollector.computed_ytest)

            result.config(text='Results from LGMLVQ Classification:' + '\n' + '\n' + str('Training Accuracy=') + str(train_score) + 
                            '\n' + str('Test Accuracy=') + str(test_score))            
            
            pred = lgmlvq.predict(inputscollector.computed_Xtest) 
            inputscollector.predictions = pred

        trainmodel = ttk.Button(popup, text='Train', command = lgmlvq_train)
        trainmodel.pack(padx=10, pady=10, side='top')

        def get_folds():
            global foldentry_3

            popfold = tk.Tk()
            popfold.wm_title("No of folds")
            flds = Label(popfold, text='Enter no of folds for Cross val:', font=NORM_FONT).pack(side='top', padx=10, pady=10)
            foldentry_3 = makeentry(popfold, "No of folds:", 10)
            foldentry_3.insert(0,'5')
            foldentry_3.focus_set()

            fldsbot = ttk.Button(popfold, text='Compute', command = cross_validate_lgmlvq).pack(side='top', padx=10, pady=10)
            popfold.destroy

        def cross_validate_lgmlvq():

            lgmlvq = LgmlvqModel(prototypes_per_class=int(p_entry.get()))
            clf  = lgmlvq.fit(inputscollector.computed_Xtrain, inputscollector.computed_ytrain)

            scores = cross_val_score(clf, inputscollector.computed_Xtest, inputscollector.computed_ytest, cv=int(foldentry_3.get()))
            mean_score = np.mean(scores).astype(str)
            cvscores.config(text='--------------------------------\n' + 'Cross Validations:' + '\n' + '\n' + str('Average Cross Validation score = ') + str(mean_score)) 

        crossval = ttk.Button(popup, text='Cross Validate', command = get_folds)
        crossval.pack(padx=10, pady=10)                        

        def lgmlvq_plot():
            lgmlvq = LgmlvqModel(prototypes_per_class=int(p_entry.get()))
            lgmlvq.fit(inputscollector.computed_Xtrain, inputscollector.computed_ytrain)            
            plot2d(lgmlvq, inputscollector.computed_Xtest, inputscollector.computed_ytest, 1, 'LGMLVQ')
            plt.grid()
            plt.show()         

        r = ttk.Button(popup, text='Show plots', command = lgmlvq_plot)
        r.pack(padx=10, pady=10, side='bottom')

        def confmat():
            cm = confusion_matrix(inputscollector.computed_ytest, inputscollector.predictions)
            plot_confusion_matrix(cm, classes = inputscollector.selected_labelnames, mtd='FC_', title='Confusion matrix_LGMLVQ')

        cmbutton = ttk.Button(popup, text='Show Confusion Matrix', command = confmat)
        cmbutton.pack(padx=10, pady=10, side='bottom')

        result = Label(popup, text='Results from LGMLVQ Classification:', font=NORM_FONT)
        result.pack(side='top', padx=10, pady=10)

        cvscores = Label(popup, text='--------------------------------\n' + 'Cross Validations:', font=NORM_FONT)
        cvscores.pack(side='top', padx=10, pady=10)                        

        def validate():
            result.config(state=(NORMAL if P else DISABLED))
            prt.config(state=(NORMAL if P else DISABLED))
            cvscores.config(state=(NORMAL if P else DISABLED))
            return True

     
    popup.geometry("550x570")
    popup.mainloop()


def selectdata(the_dataset):
    # this creates a pop-up window
    popup = tk.Tk() 
    popup.wm_title("Datasets")  

    # conditional loop for multiple datasets
    if the_dataset == 'iris':
        data = datasets.load_iris()
        
        # update the selected data to the class inputscollector
        inputscollector.selected_data = data['data']
        inputscollector.selected_labels = data['target']
        inputscollector.selected_labelnames = data['target_names']

        # create a label that prints the features of the dataset onto the GUI window. 
        label = ttk.Label(popup, text='Iris Plants Database' + '\n'+ '\n' +
            "Features:" + '\n'+ str(data['feature_names']) + '\n'+  '\n'+ "No. Data Samples:"+ '\n'+ str(data['data'].shape[0])+ '\n'+ '\n'+ 
                "Labels:" + str(data['target_names']), font=NORM_FONT)
        label.pack(pady=10,padx=10,side="top")

    # loop for random dataset. 
    if the_dataset == 'random':
        # pop up for sample size window
        ss = ttk.Label(popup, text='Select the sample size required. \n (in decimals)', font=NORM_FONT)
        ss.pack(pady=10, padx=10, side='top')

        sentry = Label(popup, text = 'Enter Samples per feature:', font=NORM_FONT)
        sentry.pack(pady=10, padx=10, side='top')

        # entry label for the samples. 
        s_text = StringVar()
        sentry = ttk.Entry(popup, width=15, textvariable=s_text)
        sentry.pack(pady=10, padx=10, side='top')
        # default sample size as 200. 
        sentry.insert(0,'200')
        inputscollector.gets = sentry

        def user_entry():
            # get the entry from get() function
            print('Samples selected:', sentry.get())
            #sample_data = np.append(np.random.randn(int(sentry.get()),3),np.random.randn(int(sentry.get()),3), axis=0)
            sample_label = np.append(np.zeros(int(sentry.get())), np.ones(int(sentry.get())), axis=0)
            sample_data = np.append(np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], size=int(sentry.get())), np.random.multivariate_normal([1, 0], [[1, 0], [0, 1]], size=int(sentry.get())), axis=0)

            # add to the collector class
            inputscollector.selected_data = sample_data
            inputscollector.selected_labels = sample_label
            inputscollector.selected_labelnames = np.unique(sample_label)

            # print data dimension to the gui window
            label.config(text='Data Dimension:' + '\n' + '\n' + str(sample_data.shape) + '\n' + '\n' + str("Data Labels: ") + str(np.unique(sample_label)))

        samplebutton = ttk.Button(popup, text='Save and compute', command = user_entry)
        samplebutton.pack(padx=10, pady=10, side='top')

        label = ttk.Label(popup, text="Data Dimension:", font=NORM_FONT)
        label.pack(pady=10,padx=10,side="top")

        # validate the entry for display in the gui
        def validate():
        	label.config(state=(NORMAL if P else DISABLED))
        	return True 

    # loop for wine dataset
    if the_dataset == 'wine':
        data = datasets.load_wine()
        inputscollector.selected_data = data['data']
        inputscollector.selected_labels = data['target']
        inputscollector.selected_labelnames = data['target_names']

        label = ttk.Label(popup, text='Imported Wine Dataset' + '\n'+ '\n' +
            "Features:" + '\n'+ '\n'+ str(data['feature_names']) + '\n'+ '\n'+ "Data Samples:"+ '\n'+ str(data['data'].shape[0])+ '\n'+ '\n'+ 
                "Labels:" + str(data['target_names']), wraplength=500, justify=LEFT, font=NORM_FONT)
        label.pack(pady=10,padx=10,side="top")

    # loop for wine digits
    if the_dataset == 'digits':
        data = datasets.load_digits()
        inputscollector.selected_data = data['data']
        inputscollector.selected_labels = data['target']
        inputscollector.selected_labelnames = data['target_names']

        label = ttk.Label(popup, text='Optical Recognition of Handwritten Digits Data Set' + '\n'+ '\n' +
            "Digit images:" + '\n'+ str(data['images'].shape) + '\n'+ '\n'+ "Data Set Dimension:"+ '\n'+ str(data['data'].shape)+ '\n'+ '\n'+ 
                "Labels:" + '\n' + str(data['target_names']), wraplength=500, justify=LEFT, font=NORM_FONT)
        label.pack(pady=10,padx=10,side="top")

    # if the_dataset == 'otherdata':
    #     csv_file_path = askopenfilename()
    #     print(csv_file_path)
    #     data = pd.read_csv(csv_file_path)



    B1 = ttk.Button(popup, text="Okay", command = popup.destroy)
    B1.pack(side="top")

    popup.geometry("650x400")
    popup.mainloop()

    return selected_data

# function to normalize the dataset
def feature_normalize():
    # call the selected data
    selected_data = inputscollector.selected_data
    # compute mean and variance 
    mu = np.mean(selected_data, axis = 0)
    sigma = np.std(selected_data, axis = 0)
    # normalize
    normalized =  (selected_data - mu)/sigma
    #print(normalized)

    # pop up window to display the message
    p_ = tk.Tk()
    p_.wm_title("Message")
    msg = ttk.Label(p_, text='Standard Normalization applied', font=NORM_FONT).pack(side='top', padx=10, pady=10)
    B1 = ttk.Button(p_, text="Okay", command = p_.destroy)
    B1.pack(side='top')
    p_.geometry("300x100")
    p_.mainloop() 

    return normalized

# function to get the  percentage of test set and split data into train and test.  
def percents():
    global pentry
    global p_popup

    p_popup = tk.Tk()
    p_popup.wm_title("Enter split percent")

    # label for percent entry header
    sp = ttk.Label(p_popup, text='Enter percent of data to be tested. \n (in decimals)', font=NORM_FONT)
    sp.grid(row=0, column=0, pady=10, padx=10)

    # enter the percentage
    spentry = Label(p_popup, text = 'Test percent: (enter in decimals)', font=NORM_FONT)
    spentry.grid(row=1, column=0, sticky=W)

    # get the entered percent entry
    p_text = StringVar()
    pentry = ttk.Entry(p_popup, width=15, textvariable=p_text)
    pentry.grid(columnspan=1, row=1, column=1)
    # default as 0.3%
    pentry.insert(0,'0.3')
    # add to collector 
    inputscollector.getp = pentry

    def userentry_1():
        with open(creds_1, 'w') as f:
            f.write(pentry.get())
            f.close()
        print("Data used for testing ", pentry.get())

        # recall the selected data and labels tp split it
        data = inputscollector.selected_data
        print("Original size:", data.shape[0])
        labels = inputscollector.selected_labels

        # split into train and test 
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=float(pentry.get()), stratify=labels)
        # add the splited data sets into the collector class
        inputscollector.computed_Xtrain = X_train
        inputscollector.computed_ytrain = y_train
        inputscollector.computed_Xtest = X_test
        inputscollector.computed_ytest = y_test
        print("Training Size: ", X_train.shape[0], "Test Size: ", X_test.shape[0])

        spl.config(text='Split results:' + '\n' + '\n' + str('Original size:') + str(data.shape[0]) + 
                                '\n' + str('Training size:') + str(X_train.shape[0]) + '\n' + str('Test Size:') + str(X_test.shape[0]))

    # button to split which calls the above function
    splitbutton = ttk.Button(p_popup, text='Split', command = userentry_1)
    splitbutton.grid(row=2, column=1, sticky=W, pady=10, padx=10)

    # label to show the results
    spl = Label(p_popup, text='Split results:', font=NORM_FONT)
    spl.grid(row=3, column=1, sticky=W, padx=10, pady=10)   

    # validate to show the results in the GUI
    def validate():
        spl.config(state=(NORMAL if P else DISABLED))
        return True 

    # submit button to close.
    submitbutton = ttk.Button(p_popup, text='Ok', command = p_popup.destroy)
    submitbutton.grid(row=8, column=1, pady=10, padx=10)

    p_popup.geometry("550x300")
    p_popup.mainloop()

# =========================================================================================================
class GUI_app(tk.Tk):

    def __init__(self, *args, **kwargs):
        
        tk.Tk.__init__(self, *args, **kwargs)

        # add logo and title to the GUI
        tk.Tk.iconbitmap(self, default="logo_hsmw.ico")
        tk.Tk.wm_title(self, "Classificaiton GUI")
        
        # main frame 
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand = True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # defining the menubar inside the container frame and then add menus to it
        menubar = tk.Menu(container)

        # First menu
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=quit)
        menubar.add_cascade(label="File", menu=filemenu)

        # Second Menu
        datasets = tk.Menu(menubar, tearoff=1)
        datasets.add_command(label="IRIS", command=lambda: selectdata('iris'))
        datasets.add_command(label="Wine", command=lambda: selectdata('wine'))
        datasets.add_command(label="Digits", command=lambda: selectdata('digits'))
        datasets.add_command(label="Random Dataset", command=lambda: selectdata('random'))      
        datasets.add_command(label="Import other CSV", command=import_csv_data)      
        menubar.add_cascade(label="Datasets", menu=datasets)

        # Third Menu
        prepross = tk.Menu(menubar, tearoff=2)
        menubar.add_cascade(label="Pre-Processing", menu=prepross)
        prepross.add_command(label="Normalize", command=feature_normalize)
        prepross.add_command(label="Train Test Split", command = percents)

       
        tk.Tk.config(self, menu=menubar)

        self.frames = {}
        # a small loop to add multiple pages if required later. 
        # looping through start page and plots page
        for F in (StartPage, plots_page):

            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()
        
# make a start page which is inside the Title page 
class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)
        # make a header label
        label = tk.Label(self, text=("""Learning Vector Quantization toolkit """), font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        label1 = tk.Label(self,compound = tk.CENTER, text=explanation + '\n'+ explain2, wraplength=500, justify=LEFT, font=NORM_FONT)
        label1.pack(pady=10,padx=10)

        # defining the main buttons for model computation. 
        button1 = ttk.Button(self, text="Train with GLVQ", width=20, command=lambda: before_train_msg("GLVQ"))
        button1.pack(side='top', padx=15, pady=15)

        button2 = ttk.Button(self, text="Train with GMLVQ", width=20, command=lambda: before_train_msg("GMLVQ"))
        button2.pack(side='top', padx=15, pady=15)

        button3 = ttk.Button(self, text="Train with GRLVQ", width=20, command=lambda: before_train_msg("GRLVQ"))
        button3.pack(side='top', padx=15, pady=15)

        button4 = ttk.Button(self, text="Train with LGMLVQ", width=20, command=lambda: before_train_msg("LGMLVQ"))
        button4.pack(side='top', padx=15, pady=15)

        #button = ttk.Button(self, text="Show Results", width=20, command=lambda: controller.show_frame(plots_page))
        #button.pack(side='top',padx=15, pady=15)

        footnote = Label(self, text='GUI implementation:' + '\n' + 'Venkatramani R' + '\n' 'HSMW: Msc. Applied Mathematics', font=SMALL_FONT)
        footnote.pack(side='right')

# This is an exmaple for a addtional plots page (Only for further development if required)
# add plots_page to the loop
class plots_page(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Results", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        # this returns to home page
        button1 = ttk.Button(self, text="Back to Home page", command=lambda: controller.show_frame(StartPage))
        button1.pack()
                

app = GUI_app()
app.geometry("600x450")
app.mainloop()
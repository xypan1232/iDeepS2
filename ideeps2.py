import sys
import os
import numpy
import pdb
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
#from keras.layers import Input, merge, LSTM
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils
from keras.optimizers import SGD, RMSprop, Adadelta, Adagrad, Adam
#from keras.engine.topology import Layer
from keras.layers import normalization
#from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import LSTM, Bidirectional, Layer 
from keras.layers.embeddings import Embedding
#from keras.layers.convolutional import MaxPooling2D,Convolution1D, MaxPooling1D
from keras.layers import Conv1D, MaxPooling1D
from keras import regularizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.constraints import maxnorm
from keras.models import load_model
#from keras import initializations
#from seya.layers.recurrent import Bidirectional
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
#from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.externals import joblib 
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
import random
from random import choice
import gzip
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics import roc_auc_score
#from sklearn.cross_validation import train_test_split
#from sklearn.grid_search import GridSearchCV
from scipy import sparse
import pdb
from math import  sqrt
from sklearn.metrics import roc_curve, auc
import theano
import subprocess as sp
import scipy.stats as stats
#from seq_motifs import *
#import structure_motifs
from keras import backend as K
from rnashape_structure import run_rnashape
import argparse
import utils as utils
#from Motif import Motif
from One_Hot_Encoder import One_Hot_Encoder
from Alphabet_Encoder import Alphabet_Encoder
import re

def calculate_performace(test_num, pred_y,  labels):
    tp =0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] ==1:
            if labels[index] == pred_y[index]:
                tp = tp +1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn +1
            else:
                fp = fp + 1               
            
    acc = float(tp + tn)/test_num
    precision = float(tp)/(tp+ fp)
    sensitivity = float(tp)/ (tp+fn)
    specificity = float(tn)/(tn + fp)
    MCC = float(tp*tn-fp*fn)/(np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
    return acc, precision, sensitivity, specificity, MCC 

def focal_loss(gamma=2, alpha=2):
    def focal_loss_fixed(y_true, y_pred):
        if(K.backend()=="tensorflow"):
            import tensorflow as tf
            pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
            return -K.sum(alpha * K.pow(1. - pt, gamma) * K.log(pt))
        if(K.backend()=="theano"):
            import theano.tensor as T
            pt = T.where(T.eq(y_true, 1), y_pred, 1 - y_pred)
            return -K.sum(alpha * K.pow(1. - pt, gamma) * K.log(pt))
    return focal_loss_fixed

def read_seq(seq_file):
    seq_list = []
    seq = ''
    with gzip.open(seq_file, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name = line[1:-1]
                if len(seq):
                    seq_array = get_RNA_seq_concolutional_array(seq)
                    seq_list.append(seq_array)                    
                seq = ''
            else:
                seq = seq + line[:-1]
        if len(seq):
            seq_array = get_RNA_seq_concolutional_array(seq)
            seq_list.append(seq_array) 
    
    return np.array(seq_list)

def load_label_seq(seq_file):
    label_list = []
    seq = ''
    with gzip.open(seq_file, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name = line[1:-1]
                posi_label = name.split(';')[-1]
                label = posi_label.split(':')[-1]
                label_list.append(int(label))
    return np.array(label_list)

def read_rnashape(structure_file):
    struct_dict = {}
    index = 0
    with gzip.open(structure_file, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name = line[:-1]
            else:
                strucure = line[:-1]
                struct_dict[name] = strucure
                
    return struct_dict

def run_rnastrcutre(seq):
    #print 'running rnashapes'
    seq = seq.replace('T', 'U')
    struc_en = run_rnashape(seq)
    #fw.write(struc_en + '\n')
    return struc_en

def read_seq_graphprot(seq_file, label = 1):
    seq_list = []
    labels = []
    seq = ''
    with open(seq_file, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name = line[1:-1]
            else:
                seq = line[:-1].upper().replace('T', 'U')
                if len(seq) > 301:
                    gap_len = (len(seq) - 301)/2
                    seq = seq[gap_len:301 + gap_len]
                seq_list.append(seq)
                labels.append(label)

    return seq_list, labels

def read_seq_graphprot_test(seq_file):
    seq_list = []
    labels = []
    seq = ''
    with open(seq_file, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name = line[1:-1]
            else:
                seq = line[:-1].upper().replace('T', 'U')
                if len(seq) > 301:
                    gap_len = (len(seq) - 301)/2
                    seq = seq[gap_len:301 + gap_len]
                seq_list.append(seq)
                labels.append(name)

    return seq_list, labels

def read_seq_flanking_test(seq_file):
    seq_list = []
    labels = []
    seq = ''
    with open(seq_file, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name = line[1:-1]
            else:
                seq = line[:-1].upper().replace('T', 'U')
                seq_list.append(seq)
                labels.append(name)

    return seq_list, labels

def loaddata_graphprot(protein, path = '/home/panxy/eclipse/ideep/circrnadb/CLIPdb.human/'):
    data = dict()
    tmp = []
    listfiles = [protein + '.positive.fa', protein + '.negative.fa']
    
    mix_label = []
    mix_seq = []
    mix_structure = []    
    for tmpfile in listfiles:
        if 'positive' in tmpfile:
            label = 1
        else:
            label = 0
        seqs, labels = read_seq_graphprot(os.path.join(path, tmpfile), label = label)
        #pdb.set_trace()
        mix_label = mix_label + labels
        mix_seq = mix_seq + seqs
        
    one_hot_matrix = get_one_hot(mix_seq)
    data["seq"] = one_hot_matrix
    data["Y"] = np.array(mix_label)
    
    return data

def padding_sequence_new(seq, max_len = 200, repkey = 'N'):
    seq_len = len(seq)
    new_seq = seq
    if seq_len < max_len:
        gap_len = max_len -seq_len
        new_seq = seq + repkey * gap_len
    return new_seq

def split_overlap_seq(seq, window_size = 200):
    overlap_size = 20
    # pdb.set_trace()
    bag_seqs = []
    seq_len = len(seq)
    if seq_len >= window_size:
        num_ins = (seq_len - window_size) / (window_size - overlap_size) + 1
        remain_ins = (seq_len - window_size) % (window_size - overlap_size)
    else:
        num_ins = 0
    bag = []
    end = 0
    for ind in range(num_ins):
        start = end - overlap_size
        if start < 0:
            start = 0
        end = start + window_size
        subseq = seq[start:end]
        bag_seqs.append(subseq)
    if num_ins == 0:
        seq1 = seq
        pad_seq = padding_sequence_new(seq1)
        bag_seqs.append(pad_seq)
    else:
        if remain_ins > 10:
            # pdb.set_trace()
            # start = len(seq) -window_size
            seq1 = seq[-window_size:]
            pad_seq = padding_sequence_new(seq1, max_len=window_size)
            bag_seqs.append(pad_seq)
    return bag_seqs

def loaddata_test(seq_file, junction = True):
    data = dict()
    tmp = []
    #listfiles = [protein + '.positive.fa', protein + '.negative.fa']

    mix_label = []
    mix_seq = []
    mix_structure = []
    label = 0
    if junction:
        mix_seq, names = read_seq_graphprot_test(seq_file)
    else:
        seqs, names = read_seq_flanking_test(seq_file)
        for seq in seqs:
            bag_seqs = split_overlap_seq(seq)
            for seq in bag_seqs:
                pad_seq = padding_sequence_new(seq, max_len = 301)
                mix_seq.append(pad_seq)

    #print 'bag_seqs', len(bag_seqs)
    # pdb.set_trace()
    #mix_label = mix_label + labels
    #mix_seq = mix_seq + seqs

    one_hot_matrix = get_one_hot(mix_seq)
    data["seq"] = one_hot_matrix
    data["Y"] = names

    return data

def read_structure(seq_list):
    structure_list = []
    for seq in seq_list:

        structure = run_rnastrcutre(seq)
        #seq_array, struct = get_RNA_structure_concolutional_array(seq, fw)
        structure_list.append(structure)
    return structure_list
        


def get_one_hot(seqs, max_len = 301):
    data = []
    structures = read_structure(seqs)
    alpha_coder = Alphabet_Encoder('ACGU', 'FTIHMS')
    alphabet = alpha_coder.alphabet
    one_hot_encoder = One_Hot_Encoder(alphabet)
    replacer_seq = lambda x: choice(alpha_coder.alph0)
    #replacer_struct = lambda x: choice(alpha_coder.alph1)
    for seq, structure in zip(seqs, structures):
        #tmp = []
        seq = seq.replace('T', 'U')
        sequence = re.sub(r"[NYMRWK]", replacer_seq, seq)
        #structure = re.sub(r"[FT]", replacer_struct, structure)
        joined = alpha_coder.encode((sequence, structure))
        one_hot_matrix = one_hot_encoder.encode(joined)
        #tmp.append(one_hot_matrix)
        
        seq_len = len(seq)
        #pdb.set_trace()
        if  seq_len < max_len:
            equal_ele = np.array([float(1)/24] * 24)
            embed_len = max_len - seq_len
            tmp = np.tile(equal_ele, (embed_len, 1))
            #tmp = []
            #for i in range(embed_len):
            #    tmp.append(equal_ele) 
            #tmp = np.array(tmp)
            one_hot_matrix = np.concatenate((one_hot_matrix, tmp), axis=0)      
        else:
            one_hot_matrix = one_hot_matrix[:max_len]
        data.append(one_hot_matrix)
        
    return np.array(data)

def load_data(path, seq = True, oli = False):
    """
        Load data matrices from the specified folder.
    """

    data = dict()
    if seq: 
        tmp = []
        tmp.append(read_seq(os.path.join(path, 'sequences.fa.gz')))
        one_hot_matrix = get_one_hot(os.path.join(path, 'sequences.fa.gz'), path)
        #tmp.append(seq_onehot)
        data["seq"] = one_hot_matrix
        #data["structure"] = structure
    
    if oli: data["oli"] = read_oli_feature(os.path.join(path, 'sequences.fa.gz'))
    
    data["Y"] = load_label_seq(os.path.join(path, 'sequences.fa.gz'))
    #np.loadtxt(gzip.open(os.path.join(path,
                #                            "matrix_Response.tab.gz")),
                #                            skiprows=1)
    #data["Y"] = data["Y"].reshape((len(data["Y"]), 1))

    return data   

def complement(seq):
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    complseq = [complement[base] for base in seq]
    return complseq

def reverse_complement(seq):
    seq = list(seq)
    seq.reverse()
    return ''.join(complement(seq))

def preprocess_data(X, scaler=None, stand = False):
    if not scaler:
        if stand:
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler    

def preprocess_labels(labels, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = np_utils.to_categorical(y)
    return y, encoder

def fork (model, n=2):
    forks = []
    for i in range(n):
        f = Sequential()
        f.add (model)
        forks.append(f)
    return forks


def set_cnn_model(input_dim, input_length):
    nbfilter = 16
    model = Sequential()
    model.add(Convolution1D(input_dim=input_dim,input_length=input_length,
                            nb_filter=nbfilter,
                            filter_length=10,
                            border_mode="valid",
                            #activation="relu",
                            subsample_length=1))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_length=3))

    model.add(Dropout(0.5))

    return model

def get_cnn_network():
    '''
     get_feature = theano.function([origin_model.layers[0].input],origin_model.layers[11].get_output(train=False),allow_input_downcast=False)
    feature = get_feature(data)
    '''
    nbfilter = 16
    print 'configure cnn network'

    seq_model = set_cnn_model(4, 111)
    struct_model = set_cnn_model(6, 111)
    #pdb.set_trace()
    model = Sequential()
    model.add(Merge([seq_model, struct_model], mode='concat', concat_axis=1))

    model.add(Bidirectional(LSTM(2*nbfilter)))
    
    model.add(Dropout(0.10))
    
    model.add(Dense(nbfilter*2, activation='relu'))
    print model.output_shape
    
    return model

def plot_motif(self, data, subseqs):
    rnas, structs = zip(*(data.alpha_coder.decode(seq) for seq in subseqs))
    logo_rna = Motif(data.alpha_coder.alph0, sequences = rnas)
    logo_struct = Motif(data.alpha_coder.alph1, sequences = structs)
    return (logo_rna, logo_struct)
    #utils.plot_motif(logo, "{}motif_kernel_{}.png".format(folder, kernel))


def get_cnn_network_alhphabet(input_length):
    print 'configure cnn network'
    nbfilter = 16


    model = Sequential()
    #model.add(Convolution1D(input_dim=24, input_length=input_length,
    #                        nb_filter=nbfilter,
    #                        filter_length=10,
    #                        border_mode="valid",
                            #activation="relu",
    #                       subsample_length=1))
    model.add(Conv1D(nbfilter,
                 10,
                 padding='valid',
                 #activation='relu',
                 strides=1, input_shape=(101, 24)))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=3))
    #pdb.set_trace() 
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(2*nbfilter)))
    model.add(Dropout(0.5))
    model.add(Dense(2*nbfilter, activation='relu'))

    model.add(Dropout(0.25))

    
    return model
    
def get_seq_targets(protein):
    path = "./datasets/clip/%s/30000/test_sample_0" % protein
    data = load_data(path)
    seq_targets = np.array(data['Y'])
    
    seqs = []
    seq = ''
    fp = gzip.open(path +'/sequences.fa.gz')
    for line in fp:
        if line[0] == '>':
            name = line[1:-1]
            if len(seq):
                seqs.append(seq)                    
            seq = ''
        else:
            seq = seq + line[:-1].replace('T', 'U')
    if len(seq):
        seqs.append(seq) 
    fp.close()
    
    return seqs, seq_targets

def get_features():
    all_weights = []
    for layer in model.layers:
       w = layer.get_weights()
       all_weights.append(w)
       
    return all_weights

def convout1_f(X):
    # The [0] is to disable the training phase flag
    return _convout1_f([0] + [X])

def get_feature(model, X_batch, index):
    inputs = [K.learning_phase()] + [model.inputs[index]]
    _convout1_f = K.function(inputs, model.layers[0].layers[index].layers[1].output)
    activations =  _convout1_f([0] + [X_batch[index]])
    
    return activations

def get_motif(model, testing, protein, y, index = 0, dir1 = 'seq_cnn/', structure  = None):
    sfilter = model.layers[0].layers[index].layers[0].get_weights()
    filter_weights_old = np.transpose(sfilter[0][:,0,:,:], (2, 1, 0)) #sfilter[0][:,0,:,:]
    print filter_weights_old.shape
    #pdb.set_trace()
    filter_weights = []
    for x in filter_weights_old:
        #normalized, scale = preprocess_data(x)
        #normalized = normalized.T
        #normalized = normalized/normalized.sum(axis=1)[:,None]
        x = x - x.mean(axis = 0)
        filter_weights.append(x)
        
    filter_weights = np.array(filter_weights)
    #pdb.set_trace()
    filter_outs = get_feature(model, testing, index)
    #pdb.set_trace()
    
    #sample_i = np.array(random.sample(xrange(testing.shape[0]), 500))
    sample_i =0

    out_dir = dir1 + protein
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    if index == 0:    
        get_motif_fig(filter_weights, filter_outs, out_dir, protein, sample_i)
    else:
        get_structure_motif_fig(filter_weights, filter_outs, out_dir, protein, y, sample_i, structure)
    
def run_network(model, total_hid, training, testing, y, validation, val_y, protein=None, structure = None):
    model.add(Dense(2, input_shape=(total_hid,)))
    model.add(Activation('sigmoid'))
    
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    #model.compile(loss=focal_loss(gamma = 1), optimizer='rmsprop')
    #pdb.set_trace()
    print 'model training'
    #class_weight = {0 : 1., 1: 3.}
    #checkpointer = ModelCheckpoint(filepath="models/" + protein + "_bestmodel.hdf5", verbose=0, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=0)

    model.fit(training, y, batch_size=64, nb_epoch=20, verbose=0, validation_data=(validation, val_y), callbacks=[earlystopper])#, class_weight = class_weight)
    
    #pdb.set_trace()
    #get_motif(model, testing, protein, y, index = 0, dir1 = 'seq_cnn1/')
    #get_motif(model, testing, protein, y, index = 1, dir1 = 'structure_cnn1/', structure = structure)

    predictions = model.predict_proba(testing)
    #species = 'Mouse_'
    species = ''
    model.save(os.path.join('models/', species + protein  + '_model.pkl'))

    return predictions, model

    
def calculate_auc(net, hid, train, test, true_y, train_y, rf = False, validation = None, val_y = None, protein = None, structure = None):
    #print 'running network' 
    if rf:
        print 'running oli'
        #pdb.set_trace()
        predict, model = run_svm_classifier(train, train_y, test)
    else:
        predict, model = run_network(net, hid, train, test, train_y, validation, val_y, protein = protein, structure = structure)

    
    auc = roc_auc_score(true_y, predict[:,1])
    
    print "Test AUC: ", auc
    return auc, predict


def run_seq_struct_cnn_network(protein, seq = True, fw = None, oli = False, min_len = 301, data_dir = '/data/home/xpan/python/CLIP/CLIPdb.human/'):
    training_data = loaddata_graphprot(protein, path = data_dir)
    
    seq_hid = 16
    struct_hid = 16
    
    train_Y = training_data["Y"]
    print len(train_Y)
    #pdb.set_trace()
    training_indice, training_label, validation_indice, validation_label = split_training_validation(train_Y, validation_size = 0.1)

    seq_data = training_data["seq"]
    print seq_data.shape
    #pdb.set_trace()
    input_length = seq_data.shape[1]
    seq_train = seq_data[training_indice]
    cnn_validation = seq_data[validation_indice]
    #cnn_validation.append(seq_validation)
     
    training_indice, training_label, test_indice, test_label = split_training_validation(training_label)
    cnn_train = seq_train[training_indice]
    testing = seq_train[test_indice]

           
    seq_net =  get_cnn_network_alhphabet(input_length)
    seq_data = []
            
    y, encoder = preprocess_labels(training_label)
    val_y, encoder = preprocess_labels(validation_label, encoder = encoder) 
    
    print len(test_label)
    true_y = test_label.copy()
    true_y_onehot = preprocess_labels(true_y, encoder = encoder)
    #pdb.set_trace() 
    print 'predicting'    


        #structure = test_data["structure"]
    seq_auc, seq_predict = calculate_auc(seq_net, seq_hid + struct_hid, cnn_train, testing, true_y, y, validation = cnn_validation,
                                          val_y = val_y, protein = protein)
         
        
    #utils.get_performance_report(true_y_onehot[0], seq_predict)
    print str(seq_auc)
    fw.write( str(seq_auc) +'\n')

    mylabel = "\t".join(map(str, true_y))
    myprob = "\t".join(map(str, seq_predict))  
    fw.write(mylabel + '\n')
    fw.write(myprob + '\n')

def split_training_validation(classes, validation_size = 0.2, shuffle = False):
    """split sampels based on balnace classes"""
    num_samples=len(classes)
    classes=np.array(classes)
    classes_unique=np.unique(classes)
    num_classes=len(classes_unique)
    indices=np.arange(num_samples)
    #indices_folds=np.zeros([num_samples],dtype=int)
    training_indice = []
    training_label = []
    validation_indice = []
    validation_label = []
    for cl in classes_unique:
        indices_cl=indices[classes==cl]
        num_samples_cl=len(indices_cl)

        # split this class into k parts
        if shuffle:
            random.shuffle(indices_cl) # in-place shuffle
        
        # module and residual
        num_samples_each_split=int(num_samples_cl*validation_size)
        res=num_samples_cl - num_samples_each_split
        
        training_indice = training_indice + [val for val in indices_cl[num_samples_each_split:]]
        training_label = training_label + [cl] * res
        
        validation_indice = validation_indice + [val for val in indices_cl[:num_samples_each_split]]
        validation_label = validation_label + [cl]*num_samples_each_split

    training_index = np.arange(len(training_label))
    random.shuffle(training_index)
    training_indice = np.array(training_indice)[training_index]
    training_label = np.array(training_label)[training_index]
    
    validation_index = np.arange(len(validation_label))
    random.shuffle(validation_index)
    validation_indice = np.array(validation_indice)[validation_index]
    validation_label = np.array(validation_label)[validation_index]    
    
            
    return training_indice, training_label, validation_indice, validation_label        
        

def plot_roc_curve(labels, probality, legend_text, auc_tag = True):
    #fpr2, tpr2, thresholds = roc_curve(labels, pred_y)
    fpr, tpr, thresholds = roc_curve(labels, probality) #probas_[:, 1])
    roc_auc = auc(fpr, tpr)
    if auc_tag:
        rects1 = plt.plot(fpr, tpr, label=legend_text +' (AUC=%6.3f) ' %roc_auc)
    else:
        rects1 = plt.plot(fpr, tpr, label=legend_text )

def read_protein_name(filename='proteinnames'):
    protein_dict = {}
    with open(filename, 'r') as fp:
        for line in fp:
            values = line.rstrip('\r\n').split('\t')
            key_name = values[0][1:-1]
            protein_dict[key_name] = values[1]
    return protein_dict

def read_fasta_file(fasta_file):
    seq_dict = {}    
    fp = gzip.open(fasta_file, 'r')
    name = ''
    name_list = []
    for line in fp:
        line = line.rstrip()
        #distinguish header from sequence
        if line[0]=='>': #or line.startswith('>')
            #it is the header
            name = line[2:] #discarding the initial >
            name_list.append(name)
            seq_dict[name] = ''
        else:
            seq_dict[name] = seq_dict[name] + line.upper().replace('U', 'T')
    fp.close()
    
    return seq_dict, name_list


def run_predict():
    data_dir = '/data/home/xpan/python/CLIP/CLIPdb.human'
    fw = open('result_file_struct_circrna_new_left_left_again_1', 'w')
    fp = open('new_protein_list')
    new_list = []
    exit_pro = set()
    for protein in fp:
        protein = protein.split('_')[0]
        new_list.append(protein)
    
    for protein in os.listdir(data_dir):
        protein = protein.split('.')[0]
        
        if protein in new_list:
            continue
        if protein in exit_pro:
            continue
        exit_pro.add(protein)
        print protein
        fw.write(protein + '\t')

        run_seq_struct_cnn_network(protein, seq = True, fw= fw)
    fp.close()
    fw.close()
    

def run_predict_old():
    data_dir = '/data/home/xpan/python/CLIP/CLIPdb.mouse'
    fw = open('result_file_mouse_circrna', 'w')
    exit_pro = set()
    for protein in os.listdir(data_dir):
        protein = protein.split('.')[0]
        print protein
        if protein in exit_pro:
            continue
        exit_pro.add(protein)

        fw.write(protein + '\t')

        run_seq_struct_cnn_network(protein, seq = True, fw= fw, data_dir = data_dir)

    fw.close()
    
def load_data_file(inputfile, seq = True, onlytest = False):
    """
        Load data matrices from the specified folder.
    """
    path = os.path.dirname(inputfile)
    data = dict()
    if seq: 
        tmp = []
        tmp.append(read_seq(inputfile))
        seq_onehot, structure = read_structure(inputfile, path)
        tmp.append(seq_onehot)
        data["seq"] = tmp
        data["structure"] = structure
    if onlytest:
        data["Y"] = []
    else:
        data["Y"] = load_label_seq(inputfile)
        
    return data

def run_network_new(model, total_hid, training, y, validation, val_y, batch_size=50, nb_epoch=20):
    model.add(Dense(2, input_shape=(total_hid,)))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    #pdb.set_trace()
    print 'model training'

    earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=0)

    model.fit(training, y, batch_size=batch_size, nb_epoch=nb_epoch, verbose=0, validation_data=(validation, val_y), callbacks=[earlystopper])

    return model

def test_ideeps2(data_file = '../CLIP/circrna/human.cir.junction.fa', model_dir = '/data/home/xpan/python/iDeepS/models', outfile='prediction.txt', junction = True):
    test_data = loaddata_test(data_file, junction = junction)
    print len(test_data)
    names = test_data["Y"]
    
    print 'predicting'    

    testing = test_data["seq"] #it includes one-hot encoding sequence and structure
    fw = open(outfile, 'w')
    for mod_n in os.listdir(model_dir):
        print mod_n
        if 'Mouse' in mod_n:
            species = 'Mouse'
            protein = mod_n.split('_')[1]
            #continue
        else:
            species = 'Human'
            protein = mod_n.split('_')[0]
            #continue
        model = load_model(os.path.join(model_dir,mod_n))

        pred = model.predict_proba(testing)[:, 1]

        #pdb.set_trace()
        if not junction:
            pred = np.split(pred, len(names))
        for rna, prob in zip(names, pred):
            if junction:
                fw.write(rna + '\t' + species + '\t' + protein + '\t' + str(prob) + '\n')
            else:
                my_prob = "\t".join(map(str, prob))
                fw.write(rna + '\t' + species + '\t' + protein + '\t' + str(my_prob) + '\n')
        #myprob = "\n".join(map(str, predictions[:, 1]))
        #fw.write(myprob)

    fw.close()


def parse_arguments(parser):
    parser.add_argument('--data_file', type=str, metavar='<data_file>', help='the sequence file used for training, it contains sequences and label (0, 1) in each head of sequence.')
    parser.add_argument('--train', type=bool, default=True, help='use this option for training model')
    parser.add_argument('--model_dir', type=str, default='models', help='The directory to save the trained models for future prediction')
    parser.add_argument('--predict', type=bool, default=False,  help='Predicting the RNA-protein binding sites for your input sequences, if using train, then it will be False')
    parser.add_argument('--out_file', type=str, default='prediction.txt', help='The output file used to store the prediction probability of testing data')
    parser.add_argument('--motif', type=bool, default=True, help='Identify motifs using CNNs.')
    parser.add_argument('--motif_dir', type=str, default='motifs', help='The directory to save the identified motifs.')
    parser.add_argument('--batch_size', type=int, default=50, help='The size of a single mini-batch (default value: 50)')
    parser.add_argument('--n_epochs', type=int, default=30, help='The number of training epochs (default value: 30)')
    args = parser.parse_args()
    return args

         
if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #args = parse_arguments(parser)
    #run_predict()
    #run_predict_old()
    data_dir = '/data/home/xpan/python/CLIP/comparison/'
    for inputfile in os.listdir(data_dir):
        input_fa = data_dir + inputfile
        output_fa = data_dir + inputfile + '.out'
        print input_fa
        print output_fa
        test_ideeps2(data_file = input_fa, outfile = output_fa, junction = True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gzip
import os
import pickle
import keras.models
from itertools import groupby, repeat
from multiprocessing import Pool
from subprocess import check_output, call
from os.path import dirname
from sklearn.preprocessing import label_binarize, scale
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
import sklearn.metrics
import numpy as np
#from shutil import which
from collections import Counter
from os import remove
from PIL import Image
from tempfile import gettempdir
from math import ceil
import seaborn as sns



def save_model(model, file_path):
    """ Save a pysster.Model object.

    This function creates two files: a pickled version of the pysster.Model object and
    an hdf5 file of the actual keras model (e.g. if file_path is 'model' two files are
    created: 'model' and 'model.h5')

    Parameters
    ----------
    model : pysster.Model
        A Model object.
    
    file_path : str
        A file name.
    """
    with gzip.open(file_path, "wb") as handle:
        pickle.dump(model.params, handle, pickle.HIGHEST_PROTOCOL)
    model.model.save("{}.h5".format(file_path))


def load_model(file_path):
    """ Load a pysster.Model object.

    Parameters
    ----------
    file_path : str
       A file containing a pickled pysster.Model object (file_path.h5 must also exist, see save_model()).

    Returns
    -------
    model : pysster.Model
        A Model object.
    """
    from pysster.Model import Model
    if not os.path.exists(file_path):
        raise RuntimeError("Path not found.")
    if not os.path.exists("{}.h5".format(file_path)):
        raise RuntimeError("HDF5 file not found.")
    with gzip.open(file_path, "rb") as handle:
        params = pickle.load(handle)
    model = Model(params, None)
    model.model = keras.models.load_model("{}.h5".format(file_path))
    return model


def save_data(data, file_path):
    """ Save a pysster.Data object.

    The object will be pickled to disk.

    Parameters
    ----------
    file_path : str
        A file name.
    """
    with gzip.open(file_path, "wb") as handle:
        pickle.dump(data, handle, pickle.HIGHEST_PROTOCOL)


def load_data(file_path):
    """ Load a pysster.Data object.

    Parameters
    ----------
    file_path : str
        A file containing a pickled pysster.Data object.
    
    Returns
    -------
    data : pysster.Data
        The Data object loaded from file.
    """
    with gzip.open(file_path, "rb") as handle:
        return pickle.load(handle)


def get_handle(file_name, mode):
    if file_name[-2:] == "gz":
        return gzip.open(file_name, mode)
    return open(file_name, mode)


def parse_fasta(handle, joiner = ""):
    delimiter = lambda line: line.startswith('>')
    for is_header, block in groupby(handle, delimiter):
        if is_header:
            header = next(block)[1:].rstrip()
        else:
            yield(header, joiner.join(line.rstrip() for line in block))



def annotate_structures(input_file, output_file):
    """ Annotate secondary structure predictions with structural contexts.

    Given dot-bracket strings this function will annote every character
    as either 'H' (hairpin), 'S' (stem), 'I' (internal loop) or 'M' (multi loop). The input file
    must be a fasta formatted file and each sequence and structure must span a single line:

    '>header
    'CCCCAUAGGGG
    '((((...)))) (-3.3)

    This is the default format of RNAfold. The output file will then contain the annotated string
    as a third line:

    '>header
    'CCCCAUAGGGG
    '((((...)))) (-3.3)
    'SSSSHHHSSSS

    Parameters
    ----------
    input_file : str
        A fasta file containing secondary structure predictions.
    
    output_file : str
        A fasta file with additional structure annotations.
    """
    handle_in = get_handle(input_file, "rt")
    handle_out = get_handle(output_file, "wt")
    for header, entry in parse_fasta(handle_in, "_"):
        entry = entry.split("_")
        bg = cgb.BulgeGraph()
        bg.from_dotbracket(entry[1].split()[0])
        handle_out.write(">{}\n".format(header))
        handle_out.write("{}\n{}\n{}\n".format(entry[0], entry[1], bg.to_element_string().upper()))
    handle_in.close()
    handle_out.close()




def auROC(labels, predictions):
    fpr, tpr, thresholds = roc_curve(labels, predictions)
    return fpr, tpr, auc(fpr, tpr)


def roc_auc_per_class(labels, predictions):
    classes = list(range(max(labels)+1))
    y_true = label_binarize(labels, classes = classes)
    if len(classes) == 2:
        return [auROC(labels, predictions[:, 1])[2]] * 2
    else:
        for x in classes:
            return [auROC(y_true[:, x], predictions[:, x])[2] for x in classes]


def auPR(labels, predictions):
    precision, recall, _ = precision_recall_curve(labels, predictions)
    return precision, recall, average_precision_score(labels, predictions)


def performance_report(labels, predictions):
    classes =  list(range(labels.shape[1]))
    roc_aucs, pr_aucs  = [], []
    import pdb
    #pdb.set_trace()
    if len(classes) == 2:
        roc_aucs = [auROC(labels[:, 1], predictions[:, 1])[2]] * 2
        pr_aucs = [auPR(labels[:, 1], predictions[:, 1])[2]] * 2
        labels = label_binarize(np.argmax(labels, axis = 1), classes = classes)
    else:
        for x in classes:
            roc_aucs.append(auROC(labels[:, x], predictions[:, x])[2])
            pr_aucs.append(auPR(labels[:, x], predictions[:, x])[2])
    #if not np.isclose(np.sum(predictions, axis=1), 1).all():
        # multi-label classification
    #    y_pred = predictions > 0.5
    #    y_pred.dtype = np.uint8
    #else:
    y_pred = label_binarize(np.argmax(predictions, axis = 1), classes = classes)
    prec_recall_f1_support = precision_recall_fscore_support(labels, y_pred)
    report = np.empty((len(classes), 6))
    for x in classes:
        report[x,:] = [prec_recall_f1_support[0][x], prec_recall_f1_support[1][x],
                       prec_recall_f1_support[2][x], roc_aucs[x],
                       pr_aucs[x], prec_recall_f1_support[3][x]]
    return report


def get_performance_report(labels, predictions):
    """ Get a performance overview of a classifier.

    The report contains precision, recall, f1-score, ROC-AUC and Precision-Recall-AUC for every
    class (in a 1 vs. all approach) and weighted averages (weighted by the the number
    of sequences 'n' in each class).

    Parameters
    ----------
    labels : numpy.ndarray
        A binary matrix of shape (num sequences, num classes) containing the true labels.
    
    predictions : numpy.ndarray
        A matrix of shape (num sequences, num classes) containing predicted probabilites.
    
    Returns
    -------
    report : str
        Summary table of the above mentioned performance measurements.
    """
    classes =  list(range(labels.shape[1]))
    report = performance_report(labels, predictions)
    out = []
    out.append("             precision    recall  f1-score   roc-auc    pr-auc ")
    formatter = "{:>12}" + "{:>10.3f}" * 5 
    for x in classes:
        out.append(formatter.format("class_{}".format(x), *report[x, 0:-1]))
    out.append('\n')
    out.append(formatter.format(
        "weighted avg",
        *np.sum(report[:,0:-1] * report[:,-1, np.newaxis], axis=0)/labels.shape[0]
    ))
    out.append('\n')
    print out
    return '\n'.join(out)



def plot_roc(labels, predictions, file_path):
    """ Get ROC curves for every class.

    In the case of more than two classes the comparions will be performed in a 1 vs. all
    approach (i.e. you get one curve per class).

    Parameters
    ----------
    labels : numpy.ndarray
        A binary matrix of shape (num sequences, num classes) containing the true labels.
    
    predictions : numpy.ndarray
        A matrix of shape (num sequences, num classes) containing predicted probabilites.
    
    file_path : str
        The file the plot should be saved to.
    """
    classes = list(range(labels.shape[1]))
    colors = _get_colors(len(classes))
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (4.6666,4))
    _hide_top_right(ax)
    ax.plot([0, 1], [0, 1], color = 'black', linewidth = 1, linestyle = '--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    if len(classes) == 2:
        fpr, tpr, roc_auc = auROC(labels[:, 1], predictions[:, 1])
        label = 'AUC = {:.3f}'.format(roc_auc)
        ax.plot(fpr, tpr, linewidth = 2.2, color = colors[0], label = label)
    else:
        for x in classes:
            fpr, tpr, roc_auc = auROC(labels[:, x], predictions[:, x])
            label = 'AUC class_{} = {:.3f}'.format(x, roc_auc)
            ax.plot(fpr, tpr, linewidth = 2.2, color = colors[x], label = label)
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., framealpha=1)
    fig.savefig(file_path, bbox_inches = 'tight')
    fig.clf()
    plt.close('all')
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)


def plot_prec_recall(labels, predictions, file_path):
    """ Get Precision-Recall curves for every class.

    In the case of more than two classes the comparions will be performed in a 1 vs. rest
    approach (i.e. you get one curve per class).

    Parameters
    ----------
    labels : numpy.ndarray
        A binary matrix of shape (num sequences, num classes) containing the true labels.
    
    predictions : numpy.ndarray
        A matrix of shape (num sequences, num classes) containing predicted probabilites.
    
    file_path : str
        The file the plot should be saved to.
    """
    classes = list(range(labels.shape[1]))
    colors = _get_colors(len(classes))
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (4.6666,4))
    _hide_top_right(ax)
    ax.plot([0, 1], [0, 1], color = 'white', linewidth = 1, linestyle = '--')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    if len(classes) == 2:
        precision, recall, prec_auc = auPR(labels[:,1], predictions[:, 1])
        label = 'AUC = {:.3f}'.format(prec_auc)
        ax.plot(recall, precision, linewidth = 2.2, color = colors[0], label = label)
    else:
        for x in classes:
            precision, recall, prec_auc = auPR(labels[:, x], predictions[:, x])
            label = 'AUC class_{} = {:.3f}'.format(x, prec_auc)
            ax.plot(recall, precision, linewidth = 2.2, color = colors[x], label = label)
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., framealpha=1)
    fig.savefig(file_path, bbox_inches = 'tight')
    fig.clf()
    plt.close('all')
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)



def save_as_meme(logos, file_path):
    """ Save sequence (or structure) motifs in MEME format.

    Parameters
    ----------
    logos : [pysster.Motif]
        A list of Motif objects.

    file_path : str
        The name of the output text file.
    """
    alphabet = logos[0].alphabet
    with open(file_path, "wt") as handle:
        handle.write("MEME version 4\n\nALPHABET= {}\n\nstrands: + -\n\n".format(alphabet))
        handle.write("Background letter frequencies (from uniform background):\n")
        if alphabet == "().":
            handle.write("{} 0.33333 {} 0.33333 {} 0.33334\n".format(*alphabet))
        else:
            handle.write("{} 0.25000 {} 0.25000 {} 0.25000 {} 0.25000\n".format(*alphabet))
        for i, logo in enumerate(logos):
            pwm = logo.pwm
            handle.write("\nMOTIF motif_{} motif_{}\n\n".format(i,i))
            handle.write("letter-probability matrix: alength= {} w= {} nsites= 20 E= 0\n".format(
                len(alphabet),
                pwm.shape[0]
            ))
            for row in range(pwm.shape[0]):
                handle.write("  {}\t\n".format("\t  ".join(str(round(x, 6)) for x in pwm[row,:])))


def run_tomtom(motif_file, output_folder, database, options = None):
    """ Compare a MEME file against a database using TomTom.

    Default options string: "-min-overlap 5 -verbosity 1 -xalph -evalue -thresh 0.1"

    Parameters
    ----------
    motif_file : str
        A MEME file.
    
    output_folder : str
        The folder the TomTom output will be saved in.
    
    database : str
        A MEME file serving as the database to compare against.
    
    option : str
        Command line options passed to TomTom.
    """
    if output_folder[-1] != "/":
        output_folder += "/"
    if not os.path.isdir(output_folder):  
        os.makedirs(output_folder)
    if options == None:
        options = "-min-overlap 5 -verbosity 1 -xalph -evalue -thresh 0.1"
    command = "tomtom {}".format(options)
    call("{} -oc {} {} {}".format(command, output_folder, motif_file, database), shell = True)


def softmax(x):
    x = np.exp(x - np.max(x))
    return x / x.sum(axis = 0)

def randargmax(x):
    vals = x.max(axis=1)
    results = np.zeros(x.shape[0], dtype=np.uint64)
    for i, val in enumerate(vals):
        results[i] = np.random.choice(np.where(np.isclose(x[i,:], val))[0])
    return results




def _hide_top_right(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')


def plot_motif_summary(position_max, mean_acts, kernel, file_path):
    classes = []
    ylim_hist, ylim_mean = 0, 0
    for i, hist in enumerate(position_max):
        if len(hist) == 0:
            print("Warning: class {} did not activate kernel {}. No plots are created.".format(
                i, kernel
            ))
        else:
            classes.append(i)
            ylim_hist = max(ylim_hist, Counter(hist).most_common(1)[0][1])
            ylim_mean = max(ylim_mean, max(mean_acts[i]))
    xlim = len(mean_acts[classes[0]]) + 1
    matplotlib.rcParams.update({'font.size': 30})
    files = []
    n_per_plot = 3
    n_plots = ceil(len(classes)/n_per_plot)
    for plot_id in range(n_plots):
        classes_left = len(classes) - plot_id*n_per_plot
        classes_this_plot = min(n_per_plot, classes_left)
        fig, ax = plt.subplots(nrows = 2, 
                               ncols = classes_this_plot,
                               figsize = (19*classes_this_plot, 12))
        for class_num in range(classes_this_plot):
            # histograms
            ax.flat[class_num].hist(position_max[class_num + n_per_plot*plot_id], 
                                    bins = xlim, range = (0, xlim))
            ax.flat[class_num].set_xlabel("sequence position")
            ax.flat[class_num].set_ylabel("counts")
            ax.flat[class_num].set_ylim((0, ylim_hist))
            ax.flat[class_num].set_title("kernel {}, class_{}, (n = {})".format(
            kernel, class_num + n_per_plot*plot_id, len(position_max[class_num + n_per_plot*plot_id])
            ))
            _hide_top_right(ax.flat[class_num])
            # mean activations
            ax.flat[class_num + classes_this_plot].plot(list(range(1, xlim)),
                                                        mean_acts[class_num + n_per_plot*plot_id],
                                                        linewidth = 5.0)
            ax.flat[class_num + classes_this_plot].set_xlabel("sequence position")
            ax.flat[class_num + classes_this_plot].set_ylabel("mean activation")
            ax.flat[class_num + classes_this_plot].set_ylim((0, ylim_mean))
            _hide_top_right(ax.flat[class_num + classes_this_plot])
        plt.tight_layout()
        files.append("{}/plotsum{}.png".format(gettempdir(), plot_id))
        fig.savefig(files[-1])
        fig.clf()
        plt.close('all')
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    images = []
    for file_name in files:
        images.append(Image.open(file_name))
    combine_images(images, file_path)
    for file_name in files:
        remove(file_name)


def plot_violins(data, kernel, file_path):
    matplotlib.rcParams.update({'font.size': 15})
    num_plots = len(data)
    labels = ["class_{}".format(x) for x in range(num_plots)]
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (max(5, num_plots), 5))
    _hide_top_right(ax)
    ax.grid(axis = "y", alpha = 0.3)
    ax.set_title("Activations, kernel {}".format(kernel))
    ax.set_ylabel("max activations")
    parts = ax.violinplot(data, showmeans = True, showextrema = True)
    ax.set_ylim(ymin = 0) 
    parts['cmeans']._linewidths = [2]
    parts['cmins']._linewidths = [2]
    parts['cmaxes']._linewidths = [2]
    parts['cbars']._linewidths = [2]
    ax.get_xaxis().set_tick_params(length=0)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    plt.xticks(rotation=90)
    fig.subplots_adjust(bottom=0.2)
    plt.tight_layout()
    fig.savefig(file_path)
    fig.clf()
    plt.close('all')
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)


def plot_motif(logo, file_path):
    if isinstance(logo, tuple):
        img1, img2 = logo[0].plot(scale=0.75), logo[1].plot(scale=0.75)
        img = Image.new("RGB", (img1.size[0], img1.size[1]+img2.size[1]))
        img.paste(img1, (0, 0))
        img.paste(img2, (0, img1.size[1]))
        img1.close()
        img2.close()
    else:
        img = logo.plot(scale=0.75)
    img.save(file_path)
    img.close()
    plt.close('all')


def _set_sns_context(n_kernel):
    if n_kernel <= 25:
        sns.set_context("notebook", rc={"ytick.labelsize":26})
    elif 25 < n_kernel <= 50:
        sns.set_context("notebook", rc={"ytick.labelsize":22})
    elif 50 < n_kernel <= 75:
        sns.set_context("notebook", rc={"ytick.labelsize":14})
    elif 75 < n_kernel <= 100:
        sns.set_context("notebook", rc={"ytick.labelsize":8})
    else:
        sns.set_context("notebook", rc={"ytick.labelsize":5})


def _get_colors(x):
    palette = ["hls", "Set1"][x < 10]
    return sns.color_palette(palette, x, 0.6)


def _plot_heatmap(file_path, data, class_id, classes = None):
    _set_sns_context(data.shape[1])
    n_classes = len(set(class_id))
    palette = _get_colors(n_classes)
    colors = [palette[x] for x in class_id]
    g = sns.clustermap(data = data.T, method = "ward", metric = "euclidean",
                       cmap = "RdBu_r", xticklabels = False, yticklabels = True,
                       figsize = (30,25), row_cluster = True, col_cluster = True,
                       linewidths = 0, col_colors = colors, robust = True,
                       z_score = 0, cbar_kws={"ticks":[-1.5,0,+1.5]})
    g.ax_col_dendrogram.set_xlim([0,0])
    g.ax_col_dendrogram.set_ylim([0,0])
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
    sns.set(font_scale=2.8)
    if classes == None:
        classes = list(range(n_classes))
    for x in range(n_classes):
        g.ax_col_dendrogram.bar(0, 0, color=palette[x],
                                label="class_{}".format(classes[x]), linewidth=0)
    g.ax_col_dendrogram.legend(loc = "center", ncol = min(6, n_classes))
    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize=25)
    plt.savefig(file_path, bbox_inches = 'tight')
    plt.close('all')
    sns.reset_orig()
    sns.set()
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)


def combine_images(images, output_file):
    widths, heights = zip(*(i.size for i in images))
    new_im = Image.new('RGB', (max(widths), sum(heights)), "#ffffff")
    y_offset = 0
    for im in images:
        new_im.paste(im, (0, y_offset))
        im.close()
        y_offset += im.size[1]
    new_im.save(output_file)
    new_im.close()
    plt.close('all')

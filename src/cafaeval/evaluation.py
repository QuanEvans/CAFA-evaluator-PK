import os
import numpy as np
import pandas as pd
import multiprocessing as mp
from dataclasses import dataclass
from cafaeval.parser import obo_parser, gt_parser, pred_parser, gt_exclude_parser, update_toi
from cafaeval.tests import test_norm_metric, test_intersection
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())


# Return a mask for all the predictions (matrix) >= tau
def solidify_prediction(pred, tau):
    return pred >= tau


# computes the f metric for each precision and recall in the input arrays
def compute_f(pr, rc):
    n = 2 * pr * rc
    d = pr + rc
    return np.divide(n, d, out=np.zeros_like(n, dtype=float), where=d != 0)


def compute_s(ru, mi):
    return np.sqrt(ru**2 + mi**2)
    # return np.where(np.isnan(ru), mi, np.sqrt(ru + np.nan_to_num(mi)))


def compute_confusion_matrix(tau_arr, g, pred_matrix, toi, n_gt, ic_arr=None):
    """
    Perform the evaluation at the matrix level for all tau thresholds
    The calculation is
    """
    # n, tp, fp, fn, pr, rc (fp = misinformation, fn = remaining uncertainty)
    metrics = np.zeros((len(tau_arr), 6), dtype='float')

    for i, tau in enumerate(tau_arr):

        # Filter predictions based on tau threshold
        p = solidify_prediction(pred_matrix, tau)

        # Terms subsets
        intersection = np.logical_and(p, g)  # TP
        mis = np.logical_and(p, np.logical_not(g))  # FP, predicted but not in the ground truth
        remaining = np.logical_and(np.logical_not(p), g)  # FN, not predicted but in the ground truth

        # Weighted evaluation
        if ic_arr is not None:
            p = p * ic_arr[toi]
            intersection = intersection * ic_arr[toi]  # TP
            mis = mis * ic_arr[toi]  # FP, predicted but not in the ground truth
            remaining = remaining * ic_arr[toi]  # FN, not predicted but in the ground truth

        n_pred = p.sum(axis=1)  # TP + FP (number of terms predicted in each protein)
        n_intersection = intersection.sum(axis=1)  # TP (number of TP terms per protein)
        # Number of proteins with at least one term predicted with score >= tau
        metrics[i, 0] = (p.sum(axis=1) > 0).sum()

        # Sum of confusion matrices
        metrics[i, 1] = n_intersection.sum()  # TP (total terms)
        metrics[i, 2] = mis.sum(axis=1).sum()  # FP
        metrics[i, 3] = remaining.sum(axis=1).sum()  # FN

        # Macro-averaging
        metrics[i, 4] = np.divide(n_intersection, n_pred, out=np.zeros_like(n_intersection, dtype='float'), where=n_pred > 0).sum()  # Precision
        metrics[i, 5] = np.divide(n_intersection, n_gt, out=np.zeros_like(n_gt, dtype='float'), where=n_gt > 0).sum()  # Recall

    return metrics


def compute_confusion_matrix_exclude(tau_arr, g_perprotein, pred_matrix, toi_perprotein, n_gt, ic_arr=None):
    """
    Perform the evaluation at the matrix level for all tau thresholds
    The calculation is

    Here, g is the full ground truth matrix without filtering terms of interest (toi).
    Instead,
    """
    # n, tp, fp, fn, pr, rc (fp = misinformation, fn = remaining uncertainty)
    metrics = np.zeros((len(tau_arr), 6), dtype='float')

    for i, tau in enumerate(tau_arr):

        # Filter predictions based on tau threshold
        p_perprotein = [solidify_prediction(pred_matrix[p_idx, tois], tau) for p_idx, tois in enumerate(toi_perprotein)]

        # Terms subsets
        intersection = [np.logical_and(p_i, g_i) for p_i, g_i in zip(p_perprotein, g_perprotein)]  # TP
        mis = [np.logical_and(p_i, np.logical_not(g_i)) for p_i, g_i in zip(p_perprotein, g_perprotein)]  # FP, predicted but not in the ground truth
        remaining = [np.logical_and(np.logical_not(p_i), g_i) for p_i, g_i in zip(p_perprotein, g_perprotein)]  # FN, not predicted but in the ground truth

        # Weighted evaluation
        if ic_arr is not None:
            p_perprotein = [p_i * ic_arr[tois] for p_i, tois in zip(p_perprotein, toi_perprotein)]
            intersection = [inter * ic_arr[tois] for inter, tois in zip(intersection, toi_perprotein)]  # TP
            mis = [misinf * ic_arr[tois] for misinf, tois in zip(mis, toi_perprotein)]  # FP, predicted but not in the ground truth
            remaining = [rem * ic_arr[tois] for rem, tois in zip(remaining, toi_perprotein)]  # FN, not predicted but in the ground truth

        n_pred = np.array([p_i.sum() for p_i in p_perprotein])  # TP + FP
        n_intersection = np.array([inter.sum() for inter in intersection])  # TP
        precision = np.divide(n_intersection, n_pred, out=np.zeros_like(n_intersection, dtype='float'), where=n_pred > 0)
        recall = np.divide(n_intersection, n_gt, out=np.zeros_like(n_gt, dtype='float'), where=n_gt > 0)

        # metrics tests
        test_norm_metric(precision, name='precision')
        test_norm_metric(recall, name='recall')
        test_intersection(n_intersection, n_pred, n_gt)


        # Number of proteins with at least one term predicted with score >= tau
        metrics[i, 0] = (n_pred > 0).sum()

        # Sum of confusion matrices
        metrics[i, 1] = n_intersection.sum()  # TP
        metrics[i, 2] = np.sum([m.sum() for m in mis])  # FP
        metrics[i, 3] = np.sum([r.sum() for r in remaining])  # FN

        # Macro-averaging
        metrics[i, 4] = precision.sum()  # Precision
        metrics[i, 5] = recall.sum()  # Recall

    print("metrics calculated")
    return metrics


@dataclass
class ProteinMetricData:
    scores_asc: np.ndarray
    pred_prefix: np.ndarray
    tp_prefix: np.ndarray
    gt_total: float
    size: int


def _prepare_protein_metric_data(pred_row, gt_row, weights_row, gt_total):
    """
    Build per-protein structures required to aggregate metrics across taus quickly.
    """
    mask = pred_row > 0
    scores = pred_row[mask]
    gt_flags = gt_row[mask].astype(bool)

    if weights_row is None:
        weights = None
    else:
        weights = weights_row[mask]

    if scores.size > 0:
        order_asc = np.argsort(scores)
        scores_asc = scores[order_asc]
        order_desc = order_asc[::-1]
        if weights is None:
            weights_desc = np.ones(scores.size, dtype=np.float64)
        else:
            weights_desc = weights[order_desc].astype(np.float64, copy=False)
        gt_desc = gt_flags[order_desc]
        tp_values = weights_desc * gt_desc
        pred_prefix = np.concatenate(([0.0], np.cumsum(weights_desc, dtype=np.float64)))
        tp_prefix = np.concatenate(([0.0], np.cumsum(tp_values, dtype=np.float64)))
        size = scores.size
    else:
        scores_asc = np.empty(0, dtype=np.float64)
        pred_prefix = np.zeros(1, dtype=np.float64)
        tp_prefix = np.zeros(1, dtype=np.float64)
        size = 0

    return ProteinMetricData(scores_asc=scores_asc,
                             pred_prefix=pred_prefix,
                             tp_prefix=tp_prefix,
                             gt_total=float(gt_total),
                             size=size)


def _accumulate_metrics_for_chunk(protein_chunk, tau_arr):
    n_tau = len(tau_arr)
    chunk_metrics = np.zeros((n_tau, 6), dtype=np.float64)
    if not protein_chunk:
        return chunk_metrics

    zeros_idx = np.zeros(n_tau, dtype=np.int32)
    for data in protein_chunk:
        if data.size:
            idx = np.searchsorted(data.scores_asc, tau_arr, side='left')
            counts = data.size - idx
        else:
            counts = zeros_idx

        pred = data.pred_prefix[counts]
        tp = data.tp_prefix[counts]
        fp = pred - tp
        fn = data.gt_total - tp

        has_pred = pred > 0
        chunk_metrics[:, 0] += has_pred.astype(np.float64)
        chunk_metrics[:, 1] += tp
        chunk_metrics[:, 2] += fp
        chunk_metrics[:, 3] += fn

        precision = np.divide(tp, pred, out=np.zeros_like(tp), where=has_pred)
        if data.gt_total > 0:
            recall = tp / data.gt_total
        else:
            recall = np.zeros_like(tp)
        chunk_metrics[:, 4] += precision
        chunk_metrics[:, 5] += recall

    return chunk_metrics


def _compute_metrics_sparse(protein_data, tau_arr, n_cpu):
    if not protein_data:
        return np.zeros((len(tau_arr), 6), dtype=np.float64)

    if n_cpu <= 1 or len(protein_data) < 2:
        metrics = _accumulate_metrics_for_chunk(protein_data, tau_arr)
    else:
        chunk_size = int(np.ceil(len(protein_data) / n_cpu))
        chunks = [protein_data[i:i + chunk_size] for i in range(0, len(protein_data), chunk_size)]
        with mp.Pool(processes=n_cpu) as pool:
            partials = pool.starmap(_accumulate_metrics_for_chunk, [(chunk, tau_arr) for chunk in chunks])
        metrics = np.sum(partials, axis=0)

    print("Jobs on all CPUs completed.")
    return metrics


def compute_metrics(pred, gt_matrix, tau_arr, toi, gt_exclude=None, ic_arr=None, n_cpu=0):
    """
    Takes the prediction and the ground truth and for each threshold in tau_arr
    calculates the confusion matrix and returns the coverage,
    precision, recall, remaining uncertainty and misinformation.
    Toi is the list of terms (indexes) to be considered
    """
    # Parallelization
    if n_cpu == 0:
        n_cpu = mp.cpu_count()

    columns = ["n", "tp", "fp", "fn", "pr", "rc"]
    proteins_has_gt = gt_matrix[:, toi].sum(1) > 0
    proteins_with_gt = np.where(proteins_has_gt)[0]
    gt_with_annots = gt_matrix[proteins_with_gt, :]
    pred_with_gt = pred[proteins_has_gt, :]

    if gt_exclude is not None:
        toi_perprotein = [np.setdiff1d(toi, gt_exclude.matrix[p_idx, :].nonzero()[0],
                                       assume_unique=True) for p_idx in proteins_with_gt]
        gt_perprotein = [gt_with_annots[p_idx, tois] for p_idx, tois in enumerate(toi_perprotein)]
        n_gt = np.array([gpp.sum().item() for gpp in gt_perprotein])
        if np.any(n_gt == 0):
            print(f'Proteins with no annotations in TOI {np.count_nonzero(n_gt == 0)}')
        if ic_arr is not None:
            n_gt = np.array([(gpp * ic_arr[tois]).sum().item() for gpp, tois in zip(gt_perprotein, toi_perprotein)])

        protein_data = []
        for idx, tois in enumerate(toi_perprotein):
            weights_row = ic_arr[tois] if ic_arr is not None else None
            pred_row = pred_with_gt[idx, tois]
            gt_row = gt_perprotein[idx]
            protein_data.append(_prepare_protein_metric_data(pred_row, gt_row, weights_row, n_gt[idx]))
    else:
        g = gt_with_annots[:, toi]
        p = pred_with_gt[:, toi]
        if ic_arr is None:
            n_gt = g.sum(axis=1)
            weights_row = None
        else:
            weights_row = ic_arr[toi]
            n_gt = (g * weights_row).sum(axis=1)
        protein_data = [_prepare_protein_metric_data(p_row, g_row, weights_row, gt_total)
                        for p_row, g_row, gt_total in zip(p, g, n_gt)]

    metrics = _compute_metrics_sparse(protein_data, tau_arr, n_cpu)
    return pd.DataFrame(metrics, columns=columns)


def normalize(metrics, ns, tau_arr, ne, normalization):

    # Normalize columns
    for column in metrics.columns:
        if column != "n":
            # By default normalize by gt
            denominator = ne
            # Otherwise normalize by pred
            if normalization == 'pred' or (normalization == 'cafa' and column == "pr"):
                denominator = metrics["n"]
            metrics[column] = np.divide(metrics[column], denominator,
                                        out=np.zeros_like(metrics[column], dtype='float'),
                                        where=denominator > 0)

    metrics['ns'] = [ns] * len(tau_arr)
    metrics['tau'] = tau_arr
    metrics['cov'] = metrics['n'] / ne
    metrics['mi'] = metrics['fp']
    metrics['ru'] = metrics['fn']

    metrics['f'] = compute_f(metrics['pr'], metrics['rc'])
    metrics['s'] = compute_s(metrics['ru'], metrics['mi'])

    # Micro-average, calculation is based on the average of the confusion matrices
    metrics['pr_micro'] = np.divide(metrics['tp'], metrics['tp'] + metrics['fp'],
                                    out=np.zeros_like(metrics['tp'], dtype='float'),
                                    where=(metrics['tp'] + metrics['fp']) > 0)
    metrics['rc_micro'] = np.divide(metrics['tp'], metrics['tp'] + metrics['fn'],
                                    out=np.zeros_like(metrics['tp'], dtype='float'),
                                    where=(metrics['tp'] + metrics['fn']) > 0)
    metrics['f_micro'] = compute_f(metrics['pr_micro'], metrics['rc_micro'])

    return metrics


def _add_weighted_suffix(df):
    """Rename metric columns to _w when weighted data is the only output."""
    suffix_cols = {col: f'{col}_w' for col in df.columns if col not in ('ns', 'tau')}
    return df.rename(columns=suffix_cols)


def evaluate_prediction(prediction, gt, ontologies, tau_arr, gt_exclude=None, normalization='cafa', n_cpu=0,
                        weighted_only=False):

    dfs = []
    dfs_w = []

    # Unweighted metrics
    for ns in prediction:
        # number of proteins with positive annotations
        proteins_has_gt = gt[ns].matrix[:, ontologies[ns].toi].sum(1) > 0
        proteins_with_gt = np.where(proteins_has_gt)[0]
        num_annot_prots = proteins_has_gt.sum()  # number of proteins with positive annotations in TOIs
        if gt_exclude is None:
            exclude = None
        else:
            exclude = gt_exclude[ns]
            toi_perprotein = [
                np.setdiff1d(ontologies[ns].toi, gt_exclude[ns].matrix[p, :].nonzero()[0],
                             assume_unique=True) for p in proteins_with_gt]
            # update the number of proteins with positive annotations, now on protein-specific TOIs
            num_annot_prots = sum([gt[ns].matrix[p, toi_perprotein[p_idx]].sum()>0 for
                                   p_idx, p in enumerate(proteins_with_gt)])

        if not weighted_only:
            ne = np.full(len(tau_arr), num_annot_prots)

            dfs.append(normalize(compute_metrics(
                prediction[ns].matrix, gt[ns].matrix, tau_arr, ontologies[ns].toi, exclude, None, n_cpu),
                                 ns, tau_arr, ne, normalization))

        # Weighted metrics
        if ontologies[ns].ia is not None:

            # number of proteins with positive annotations
            proteins_has_gt = gt[ns].matrix[:, ontologies[ns].toi_ia].sum(1) > 0
            num_annot_prots = (proteins_has_gt).sum()

            if gt_exclude is None:
                exclude = None
            else:
                exclude = gt_exclude[ns]
                toi_perprotein_ia = [
                    np.setdiff1d(ontologies[ns].toi_ia, gt_exclude[ns].matrix[p, :].nonzero()[0],
                                 assume_unique=True) for p in proteins_with_gt]
                # update the number of proteins with positive annotations, now on protein-specific TOIs
                num_annot_prots = sum([gt[ns].matrix[p, toi_perprotein_ia[p_idx]].sum() > 0 for
                                       p_idx, p in enumerate(proteins_with_gt)])

            ne = np.full(len(tau_arr), num_annot_prots)

            dfs_w.append(normalize(compute_metrics(
                prediction[ns].matrix, gt[ns].matrix, tau_arr, ontologies[ns].toi_ia, exclude, ontologies[ns].ia, n_cpu),
                ns, tau_arr, ne, normalization))
        elif weighted_only:
            raise ValueError(f"Weighted metrics requested but IA file missing for namespace {ns}")

    dfs = pd.concat(dfs) if dfs else None

    # Merge weighted and unweighted dataframes
    if dfs_w:
        dfs_w = pd.concat(dfs_w)
        if dfs is None:
            dfs = _add_weighted_suffix(dfs_w) if weighted_only else dfs_w
        else:
            dfs = pd.merge(dfs, dfs_w, on=['ns', 'tau'], suffixes=('', '_w'))

    return dfs


def cafa_eval(obo_file, pred_dir, gt_file, ia=None, no_orphans=False, norm='cafa', prop='max',
              exclude=None, toi_file=None, max_terms=None, th_step=0.01, n_cpu=1, weighted_only=False):

    # Tau array, used to compute metrics at different score thresholds
    tau_arr = np.arange(th_step, 1, th_step)

    # Parse the OBO file and creates a different graphs for each namespace
    ontologies = obo_parser(obo_file, ("is_a", "part_of"), ia, not no_orphans)
    if toi_file is not None:
        ontologies = update_toi(ontologies, toi_file)

    if weighted_only and ia is None:
        raise ValueError("Weighted-only evaluation requires an Information Accretion file")

    # Parse ground truth file
    gt = gt_parser(gt_file, ontologies)
    if exclude is not None:
        gt_exclude = gt_exclude_parser(exclude, gt, ontologies)
    else:
        gt_exclude = None

    # Set prediction files looking recursively in the prediction folder
    pred_folder = os.path.normpath(pred_dir) + "/"  # add the tailing "/"
    pred_files = []
    for root, dirs, files in os.walk(pred_folder):
        for file in files:
            pred_files.append(os.path.join(root, file))
    logging.debug("Prediction paths {}".format(pred_files))

    # Parse prediction files and perform evaluation
    dfs = []
    for file_name in pred_files:
        print(file_name)
        prediction = pred_parser(file_name, ontologies, gt, prop, max_terms)
        if not prediction:
            logging.warning("Prediction: {}, not evaluated".format(file_name))
        else:
            df_pred = evaluate_prediction(prediction, gt, ontologies, tau_arr, gt_exclude,
                                          normalization=norm, n_cpu=n_cpu, weighted_only=weighted_only)
            df_pred['filename'] = file_name.replace(pred_folder, '').replace('/', '_')
            dfs.append(df_pred)
            logging.info("Prediction: {}, evaluated".format(file_name))

    # Concatenate all dataframes and save them
    df = None
    dfs_best = {}
    if dfs:
        df = pd.concat(dfs)

        # Remove rows with no coverage
        coverage_col = 'cov' if 'cov' in df.columns else 'cov_w' if 'cov_w' in df.columns else None
        if coverage_col is not None:
            df = df[df[coverage_col] > 0].reset_index(drop=True)
        else:
            raise ValueError("Unable to determine coverage column in evaluation results")
        df.set_index(['filename', 'ns', 'tau'], inplace=True)

        # Calculate the best index for each namespace and each evaluation metric
        for metric, cols in [('f', ['rc', 'pr']), ('f_w', ['rc_w', 'pr_w']), ('s', ['ru', 'mi']), ('f_micro', ['rc_micro', 'pr_micro']), ('f_micro_w', ['rc_micro_w', 'pr_micro_w'])]:
            if metric in df.columns:
                index_best = df.groupby(level=['filename', 'ns'])[metric].idxmax() if metric in ['f', 'f_w', 'f_micro', 'f_micro_w'] else df.groupby(['filename', 'ns'])[metric].idxmin()
                df_best = df.loc[index_best]
                cov_col = 'cov' if metric[-2:] != '_w' else 'cov_w'
                if cov_col in df.columns:
                    df_best['cov_max'] = df.reset_index('tau').loc[[ele[:-1] for ele in index_best]].groupby(level=['filename', 'ns'])[cov_col].max()
                dfs_best[metric] = df_best
    else:
        logging.info("No predictions evaluated")

    return df, dfs_best


def write_results(df, dfs_best, out_dir='results', th_step=0.01):

    # Create output folder here in order to store the log file
    out_folder = os.path.normpath(out_dir) + "/"
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)

    # Set the number of decimals to write in the output files based on the threshold step size
    decimals = int(np.ceil(-np.log10(th_step))) + 1

    df.to_csv('{}/evaluation_all.tsv'.format(out_folder), float_format="%.{}f".format(decimals), sep="\t")

    for metric in dfs_best:
        dfs_best[metric].to_csv('{}/evaluation_best_{}.tsv'.format(out_folder, metric), float_format="%.{}f".format(decimals), sep="\t")

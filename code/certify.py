# evaluate a smoothed classifier on a dataset
import argparse
import os
import setGPU
from datasets import get_dataset, DATASETS, get_num_classes
from core import Smooth
from time import time
import torch
import datetime
from architectures import get_architecture

parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("sigma", type=float, help="noise hyperparameter")
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--method", choices=["baseline","second_order", "dipole"], default="baseline", help="cerification method")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
args = parser.parse_args()

if __name__ == "__main__":
    # load the base classifier
    checkpoint = torch.load(args.base_classifier)
    base_classifier = get_architecture(checkpoint["arch"], args.dataset)
    base_classifier.load_state_dict(checkpoint['state_dict'])

    # create the smooothed classifier g
    smoothed_classifier = Smooth(base_classifier, get_num_classes(args.dataset), args.sigma)

    # prepare output file
    f = open(args.outfile, 'w')
    if (args.method == "baseline"):
        print("idx\tlabel\tpredict\tradius\traw_count\tp_A_bar\tcorrect\ttime", file=f, flush=True)
    elif (args.method == "second_order"):
        print("idx\tlabel\tpredict\tradius\traw_count\tp_A_bar\traw_grad_norm\tgrad_norm_bound\tgrad_norm_bound_nonvacuous\tcorrect\ttime", file=f, flush=True)
    elif (args.method == "dipole"):
        print("idx\tlabel\tpredict\tradius\traw_count_symmetric\traw_count_asymmetric\tp_A_symmetric_bar\tp_A_asymmetric_bar\tcorrect\ttime", file=f, flush=True)
    # iterate through the dataset
    dataset = get_dataset(args.dataset, args.split)
    for i in range(len(dataset)):

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        (x, label) = dataset[i]

        before_time = time()
        # certify the prediction of g around x
        x = x.cuda()
        if (args.method == "baseline"):
            prediction, radius, raw_count, pABar = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch)
        elif (args.method == "second_order"):
            prediction, radius, raw_count, pABar, raw_grad_norm, grad_norm_bound, grad_norm_bound_nonvacuous = smoothed_classifier.certify_second_order(x, args.N0, args.N, args.alpha, args.batch)
        elif (args.method == "dipole"):
            prediction, radius, raw_count_symmetric, raw_count_asymmetric, p_A_symmetric_bar, p_A_asymmetric_bar = smoothed_classifier.certify_dipole(x, args.N0, args.N, args.alpha, args.batch)

        after_time = time()
        correct = int(prediction == label)

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        if (args.method == "baseline"):
            print("{}\t{}\t{}\t{:.3}\t{}\t{:.6}\t{}\t{}".format(
                i, label, prediction, radius, raw_count, pABar, correct, time_elapsed), file=f, flush=True)
        elif (args.method == "second_order"):
            print("{}\t{}\t{}\t{:.3}\t{}\t{:.6}\t{:.6}\t{:.6}\t{}\t{}\t{}".format(
                i, label, prediction, radius, raw_count, pABar,raw_grad_norm, grad_norm_bound, grad_norm_bound_nonvacuous, correct, time_elapsed), file=f, flush=True)
        elif (args.method == "dipole"):
            print("{}\t{}\t{}\t{:.3}\t{}\t{}\t{:.6}\t{:.6}\t{}\t{}".format(
                i, label, prediction, float(radius), raw_count_symmetric, raw_count_asymmetric, float(p_A_symmetric_bar), float(p_A_asymmetric_bar), correct, time_elapsed), file=f, flush=True)
    f.close()

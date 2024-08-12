import os
import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="ssfl",
        choices=["ssfl", "sup", "unsup"],
        help="mode of training",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="isic2019",
        choices=["ham10000", "rsna", "isic2019","custom"],
        help="name of dataset",
    )
    parser.add_argument(
        "--root_path",
        type=str,
        default=os.path.join("..", "dataset", "rsna", "images"),
        help="dataset root dir",
    )
    parser.add_argument(
        "--csv_file_train",
        type=str,
        default=os.path.join("..", "dataset", "rsna", "train.csv"),
        help="training set csv file",
    )
    # parser.add_argument('--csv_file_val', type=str, default='/research/pheng4/qdliu/Semi/dataset/skin/validation.csv', help='validation set csv file')
    parser.add_argument(
        "--csv_file_test",
        type=str,
        default=os.path.join("..", "dataset", "rsna", "test.csv"),
        help="testing set csv file",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=8,
        help="number of classes",
    )
    parser.add_argument("--batch_size", type=int, default=48, help="batch_size per gpu")
    parser.add_argument("--drop_rate", type=int, default=0.2, help="dropout rate")
    parser.add_argument('--pl_lr', type=float, default=0.02,help='lr of pseudo labelling')
    parser.add_argument(
        "--ema_consistency", type=int, default=1, help="whether train baseline model"
    )
    parser.add_argument(
        "--base_lr", type=float, default=2e-4, help="learning rate for baseline model"
    )
    parser.add_argument(
        "--deterministic",
        type=int,
        default=1,
        help="whether use deterministic training",
    )
    parser.add_argument(
        "--aggregator",
        type=str,
        default="average",
        choices=["average", "gba"],
        help="name of aggregator",
    )
    parser.add_argument(
        "--network",
        type=str,
        default="convnext",
        choices=["convnext", "densnet"],
        help="name of aggregator",
    )
    parser.add_argument(
        "--confuse",
        type=str,
        default="covariance",
        choices=["mean", "covariance","mean,covariance"],
        help="name of aggregator",
    )
    parser.add_argument("--seed", type=int, default=1337, help="random seed")
    parser.add_argument("--gpu", type=str, default="0", help="GPU to use")
    parser.add_argument("--local_ep", type=int, default=5, help="local epoch")
    parser.add_argument("--num_users", type=int, default=10, help="local epoch")
    parser.add_argument("--rounds", type=int, default=50, help="local epoch")
    parser.add_argument("--confidence_thresh", type=float, default=0.3, help="threshold")

    ### tune
    parser.add_argument("--resume", type=str, default=None, help="model to resume")
    parser.add_argument("--start_epoch", type=int, default=0, help="start_epoch")
    parser.add_argument("--global_step", type=int, default=0, help="global_step")
    ### costs
    parser.add_argument(
        "--label_uncertainty", type=str, default="U-Ones", help="label type"
    )
    parser.add_argument("--ema_decay", type=float, default=0.99, help="ema_decay")
    parser.add_argument("--consistency", type=float, default=1, help="consistency")
    parser.add_argument(
        "--consistency_rampup", type=float, default=30, help="consistency_rampup"
    )
    parser.add_argument('--unsup_lr', type=float, default=0.02,
                        help='lr of unsupervised clients')

    parser.add_argument("--max_grad_norm",
                        dest="max_grad_norm",
                        type=float,
                        default=5,
                        help="max gradient norm allowed (used for gradient clipping)")
    parser.add_argument(
        "--com",
        type=str,
        default="",

        help="comment",
    )
    parser.add_argument('--opt', type=str, default='adam', help='sgd or adam or adamw')
    args = parser.parse_args()

    # args.root_path = "/kaggle/input/rsna-data/rsna/images".format(args.dataset)
    # args.csv_file_train = "/kaggle/input/rsna-data/rsna/train.csv".format(args.dataset)
    # args.csv_file_test = "/kaggle/input/rsna-data/rsna/test.csv".format(args.dataset)
    
    args.root_path = os.path.join("..", "data", args.dataset, "images")
    args.csv_file_train = os.path.join("..", "data", args.dataset, "train.csv")
    args.csv_file_test = os.path.join("..", "data", args.dataset, "test.csv")

    args.rounds += 1

    match args.dataset:
        case "ham10000":
            args.num_classes = 7
        case "isic2019":
            args.num_classes = 8
        case "rsna":
            args.num_classes = 5
        case "custom":
            args.num_classes = 4

    match args.dataset:
        case "isic2019":
            args.class_names = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC", "SCC"]
        case "ham10000":
            args.class_names = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
        case "rsna":
            args.class_names = [
                "Melanoma",
                "Melanocytic nevus",
                "Basal cell carcinoma",
                "Actinic keratosis",
                "Benign keratosis",
            ]
        case "custom":
            args.class_names = ["cloudy",	"rain",	"shine",	"sunrise"]
    
    if(args.network=="convnext"):
        args.mean_used_features=192
        args.covariance_used_features= args.num_classes* args.num_classes
    else:
        args.mean_used_features=args.num_classes
        args.covariance_used_features= args.num_classes* args.num_classes


    
    return args

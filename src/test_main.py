import torch
from networks.convnext.convnextv2 import ConvNeXtV2, convnextv2_huge,convnextv2_base,convnextv2_large,convnextv2_atto,convnextv2_tiny,convnextv2_nano,convnext_pico
from networks.convnext.convnext import convnext_tiny
from options import args_parser
import numpy as np
from validation import epochVal_metrics_test
from torchvision import transforms
import os
from dataloaders import dataset
from torch.utils.data import DataLoader
import numpy as np
import torch.backends.cudnn as cudnn
import random
import pandas as pd
from networks.models import DenseNet121

args = args_parser()

checkpoint_path = os.path.join("..", "model", f"{args.dataset}_{args.mode}", "best.pth")

if __name__ == "__main__":
    checkpoint = torch.load(checkpoint_path)
    
    

    if args.network=="convnext":
        net= convnext_tiny(args.num_classes,pretrained=True, in_22k=True)
    else:
        net = DenseNet121(out_size=args.num_classes, mode=args.label_uncertainty, drop_rate=args.drop_rate)

    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        model = net.cuda()
    else:
        torch.manual_seed(args.seed)
        model = net
    if len(args.gpu.split(",")) > 1:
        model = torch.nn.DataParallel(model.cuda(), device_ids=[0, 1])
    model.load_state_dict(checkpoint["state_dict"])
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    test_dataset = dataset.CheXpertDataset(
        root_dir=args.root_path,
        csv_file=args.csv_file_test,
        transform=transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    all_preds = pd.DataFrame(columns=["thresh", "AUROC", "Accu", "Sens", "Spec", "f1","Acc"])

    thre = np.arange(0.4, 0.9, 0.02)
    thre = list(thre)
    for thresh in thre:
        print("begin", thresh)

        AUROCs,accuracy, Accus, Senss, Specs, pre, f1 = epochVal_metrics_test(
            model, test_dataloader, thresh=thresh
        )

        print(f"AUROCs: {AUROCs}")
        print(f"Accus: {Accus}")
        print(f"Senss: {Senss}")
        print(f"Specs: {Specs}")
        print(f"f1: {f1}")

        AUROC_avg = np.array(AUROCs).mean()
        Accus_avg = np.array(Accus).mean()
        Senss_avg = np.array(Senss).mean()
        Specs_avg = np.array(Specs).mean()
        f1_avg = np.array(f1).mean()

        print(f"AUROC_avg: {AUROC_avg}")
        print(f"Accus_avg: {Accus_avg}")
        print(f"Senss_avg: {Senss_avg}")
        print(f"Specs_avg: {Specs_avg}")
        print(f"f1_avg: {f1_avg}")
        print(f"Accs_avg: {accuracy}")
        print(f"{100 * '='}")

        # concat the results
        all_preds = pd.concat(
            [
                all_preds,
                pd.DataFrame(
                    {
                        "thresh": [thresh],
                        "AUROC": [AUROC_avg],
                        "Accu": [Accus_avg],
                        "Sens": [Senss_avg],
                        "Spec": [Specs_avg],
                        "f1": [f1_avg],
                        "Acc": [accuracy],
                    }
                ),
            ]
        )

    os.makedirs(os.path.join("..", "test_csv"), exist_ok=True)
    all_preds.to_csv(os.path.join("..", "test_csv", f"{args.dataset}_{args.mode}.csv"), index=False)

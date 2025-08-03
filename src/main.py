from comet_ml import start
from comet_ml.integration.pytorch import log_model
from multiscale_patch_similarity_module import MPSM
from rgb_frequency_attention_module import RFAM
import argparse, os, logging, time
from torch.utils.data import DataLoader
from torch import nn
import torch
from torch import optim
from datasets.ff import FFpp
from datasets.celeb_df import CelebDF
from datasets.dffd import DFFD
from datasets.dfdcp import DFDCP
from xception import xception
from trainer import Trainer
from transform import TwoTransform, get_augs
from utils import log_print, setup_logger, L2Loss


torch.multiprocessing.set_sharing_strategy("file_system")


def disable_batchnorm(model):
    """Disables BatchNorm layers by setting them to eval mode."""
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
            module.eval()  # Disable running mean/variance updates
            module.requires_grad_(False)

def main(args):

    hyper_params = vars(args)

    experiment = start(
        api_key="wJ5nyFWFDl079TRhZEQ6kAE5b",
        project_name="RFAM",
        workspace="mahdiasefi20"
    )
    experiment.log_parameters(hyper_params)
    save_dir = os.path.join("ckpt", args.dataset, args.exp_name, args.model_name)
    os.makedirs(save_dir, exist_ok=True)

    logfile = '{}/{}.log'.format(save_dir, time.strftime("%Y%m%d_%H%M%S", time.localtime()))
    logging.basicConfig(filename=logfile, level=logging.INFO)
    logger = logging.getLogger()

    log_print("args: {}".format(args))

    # model
    if args.model_name == "xception":
        model = xception(pretrained=True, num_classes=2)
        log_model(experiment, model=model, model_name="xception")
    else:
        raise NotImplementedError
    if torch.cuda.is_available():
        model = model.cuda()

    # transforms
    train_augs = get_augs(name=args.aug_name, norm=args.norm, size=args.size)
    # if args.consistency != "None":
    #     train_augs = TwoTransform(train_augs)
    # log_print("train aug:{}".format(train_augs))
    train_augs = get_augs(name="None", norm=args.norm, size=args.size)
    test_augs = get_augs(name="None", norm=args.norm, size=args.size)
    log_print("test aug:{}".format(test_augs))

    # dataset
    if args.dataset == "ff":
        train_dataset = FFpp(args.fake_root,
                             args.real_root, "train", train_augs, 2, args.alpha,
                             args.ff_quality)
        test_dataset = FFpp(args.fake_root,
                             args.real_root, "test", train_augs, 2, args.alpha,
                             args.ff_quality)
    elif args.dataset == "celebdf":
        train_dataset = CelebDF(args.root, "train", train_augs)
        test_dataset = CelebDF(args.root, "test", test_augs)
    elif args.dataset == "dffd":
        train_dataset = DFFD(args.root, "train", train_augs)
        test_dataset = DFFD(args.root, "test", test_augs)
    elif args.dataset == "dfdcp":
        train_dataset = DFDCP(args.root, "train", train_augs)
        test_dataset = DFDCP(args.root, "test", test_augs)
    else:
        raise NotImplementedError

    log_print("len train dataset:{}".format(len(train_dataset)))
    log_print("len test dataset:{}".format(len(test_dataset)))
    # dataloader
    trainloader = DataLoader(train_dataset,
                             batch_size=args.batch_size,
                             shuffle=args.shuffle,
                             num_workers=args.num_workers
                             )
    testloader = DataLoader(test_dataset,
                            batch_size=args.batch_size,
                            shuffle=args.shuffle,
                            num_workers=args.num_workers
                            )

    if args.num_classes == 2:
        ce_weight = [args.real_weight, 1.0]
    else:
        raise NotImplementedError

    # CrossEntropy Loss
    weight = torch.Tensor(ce_weight)
    if torch.cuda.is_available():
        weight = weight.cuda()
    ce_loss_fn = nn.CrossEntropyLoss(weight)

    # Similarity Loss (Mean Square Error)
    similarity_loss_fn = nn.MSELoss()
    # similarity_loss_fn = L2Loss()

    log_print("consistency loss function: {}, rate:{}".format(similarity_loss_fn, args.similarity_loss_rate))
    if args.optimizer == "adam":
        optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=1e-5)
    else:
        raise NotImplementedError
    log_print("optimizer: {}".format(optimizer))

    if args.load_model_path is not None:
        log_print('==> Resuming from checkpoint..')
        checkpoint = torch.load(args.load_model_path)  # , map_location="cpu"
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_recond = {
            "acc": checkpoint['acc'],
            "auc": checkpoint['auc'],
            "epoch": checkpoint['epoch'],
        }
        start_epoch = checkpoint['epoch'] + 1
        log_print("start from best recode: {}".format(best_recond))
    else:
        best_recond = {"acc": 0, "auc": 0, "epoch": -1, "tdr3": 0, "tdr4": 0}
        start_epoch = 1

    # Modules
    mspm = MPSM(k=args.k)

    rfam_low = RFAM(2 * 728)
    rfam_mid = RFAM(2 * 728)
    rfam_high = RFAM(2 * 2048)


    # trainer
    trainer = Trainer(
        train_loader=trainloader,
        test_loader=testloader,
        model=model,
        mpsm=mspm,
        rfam_low=rfam_low,
        rfam_mid=rfam_mid,
        rfam_high=rfam_high,
        optimizer=optimizer,
        ce_loss_fn=ce_loss_fn,
        similarity_loss_fn=similarity_loss_fn,
        similarity_loss_rate=args.similarity_loss_rate,
        log_interval=args.log_interval,
        best_recond=best_recond,
        save_dir=save_dir,
        exp=experiment,
        exp_name=args.exp_name)
    lr = args.lr
    for epoch_idx in range(start_epoch, args.epochs + 1):
        print("-------------------- epoch {} start-------------------------------------".format(epoch_idx))
        if epoch_idx % 10 == 0:
            trainer.optimizer = optim.Adam(params=model.parameters(), lr=lr / 2, weight_decay=1e-5)
            lr = lr / 2
            log_print("learning rate updated: {}".format(lr))
        trainer.train_epoch(epoch_idx)
        print("--------------------epoch {} end-------------------------------------".format(epoch_idx))
        print("-------------------- test epoch {} start-------------------------------------".format(epoch_idx))
        early_stop = trainer.test_epoch(epoch_idx)
        print("-------------------- test epoch {} end-------------------------------------".format(epoch_idx))
        if early_stop:
            print("-------Early Stop-------")
            break
    log_model(experiment, model=model, model_name="last_model")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--num-classes', type=int, default=2)

    # similarity loss
    # arg('--consistency', type=str, default="None")
    arg('--similarity_loss_rate', type=float, default=10)

    # transforms
    arg('--aug-name', type=str, default="None")
    arg('--norm', type=str, default="0.5")
    arg('--size', type=int, default=299)

    # dataset
    arg('--dataset', type=str, default='ff')
    arg('--ff-quality', type=str, default='c40', choices=['c23', 'c40', 'raw'])
    arg('--fake_root', type=str, default=r'/storage/users/masefi/deepFakeDetection/PCL-I2G/dataset_c40/PatchForensics/FS')
    arg('--real_root', type=str, default=r'/storage/users/masefi/deepFakeDetection/PCL-I2G/dataset_c40/PatchForensics/original')
    arg('--batch-size', type=int, default=8)
    arg('--num-workers', type=int, default=0)
    arg('--shuffle', type=bool, default=True)

    arg('--real-weight', type=float, default=4.0)

    # multiscale patch similarity module (MPSM)
    arg('--k', type=int, default=5)

    # frequency aware cue
    arg("--alpha", type=float, default=0.33)

    # optimizer
    arg('--optimizer', type=str, default="adam")
    arg('--lr', type=float, default=2e-4)

    arg('--exp-name', type=str, default='triple_sim_l2_loss')

    arg('--gpus', type=str, default='0')

    arg('--log-interval', type=int, default=1)

    arg("--epochs", type=int, default=50)
    arg("--load-model-path", type=str, default=None)
    # \\wsl.localhost\Ubuntu\home\mahdiasefi\Projects\RFAM\weights\model - data_comet - torch - model - 48.
    # pth
    arg("--model-name", type=str, default="xception")

    arg("--amp", default=False, action='store_true')

    arg("--seed", type=int, default=0)

    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    main(args)
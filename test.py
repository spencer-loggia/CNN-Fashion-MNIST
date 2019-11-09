## FF learning rate Sweep
import main
import argparse as ap

def get_args(model, lr, channels, a, b, c, d, e, f):
    """ Define our command line arguments. """
    p = ap.ArgumentParser()

    # Mode to run the model in.
    # p.add_argument("mode", choices=["train", "predict"], type=str)
    # TODO: CHANGE BACK TO ABOVE LINE!!
    # p.add_argument("--mode", type=str, default="train")

    # File locations
    p.add_argument("--data-dir", type=str, default="../release-data")
    p.add_argument("--log-file", type=str, default="../logs/best_sweep/internal_kernel/" + str(a) + "-best-logs.csv")
    p.add_argument("--model-save", type=str, default="../models/best_sweep/internal_kernel/" + str(a) + "-best-model.torch")
    p.add_argument("--predictions-file", type=str, default="../preds/best_sweep/internal_kernel/" + str(a) + "-best-preds.txt")

    # hyperparameters
    p.add_argument("--model", type=str, default=model)
    p.add_argument("--train-steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=40)
    p.add_argument("--learning-rate", type=float, default=float(lr))

    # simple-ff hparams
    p.add_argument("--ff-hunits", type=int, default=100)

    # simple-cnn hparams
    p.add_argument('--cnn-n1-channels', type=int, default=channels)
    p.add_argument('--cnn-n1-kernel', type=int, default=10)
    p.add_argument('--cnn-n2-kernel', type=int, default=5)

    # best-cnn hparams
    p.add_argument('--interior-kernel', type=int, default=a)
    p.add_argument('--initial-kernel', type=int, default=b)
    p.add_argument('--hidden', type=int, default=c)
    p.add_argument('--channel_size', type=int, default=d)
    p.add_argument('--final_channel_size', type=int, default=e)
    p.add_argument('--compression_ratio', type=int, default=f)

    return p.parse_args()


def FF_LR_sweep():
    learning_rates = ['0.000001', '0.000010', '0.000100', '0.001000', '0.010000', '0.100000', '1.000000']
    for i in learning_rates:
        args = get_args('simple-ff', i, 1)
        main.train(args)


def CNN_channel_sweep():
    channels = [1, 10, 20, 30, 35, 40, 45, 50, 60, 75, 100]
    for i in channels:
        args = get_args('simple-cnn', .001, i, 0, 0, 0, 0, 0, 0)
        main.train(args)

def best_sweep():
    init_channels = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    in_kernel = [5]
    for f in init_channels:
        print('channels: ' + str(f))
        args = get_args('best', .001, 0, 4, 3, 150, f, 0, 2)
        main.train(args)

    # for e in in_kernel:
    #     print('\n\n internal kernel ' + str(e))
    #     args = get_args('best', .001, 0, e, 5, 200,  16, 0, 2)
    #     main.train(args)

    # for a in interior_kernel:
    #     for d in chan_size:
    #         print('\n\niniterior kernel: ' + str(a) + ', channel_size: ' + str(d))
    #         args = get_args('best', .001, 0, a, 5, 200, d, 15)
    #         main.train(args)


best_sweep()


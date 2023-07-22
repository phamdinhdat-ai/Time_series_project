import argparse



def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default="transformer", help='name of model training')
    parser.add_argument('--data_type', type=str, default="dynamic", help='name of dataset training')
    parser.add_argument('--num_classes', type=int, default=5, help='numbers of classes in the dataset')
    parser.add_argument('--epochs', type=int, default=100, help='epochs')
    parser.add_argument('--sequence_length', type=int, default=None, help='sequence_length for Sequence model')
    parser.add_argument('--overlap', type=float, default=None, help='overlap of window across samples')
    parser.add_argument('--batch_size', type=int, default=128, help='setting batch_size')
    parser.add_argument('--plot', type=bool, default=True, help='plot performance')
    return parser.parse_known_args()[0] if known else parser.parse_args()


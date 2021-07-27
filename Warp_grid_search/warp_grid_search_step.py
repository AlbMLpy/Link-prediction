import argparse
import os

# Get the parameters:
parser = argparse.ArgumentParser()
parser.add_argument("--n_epoch", type=int, required=True, default=200)
parser.add_argument("--lr", type=float, required=True) # depends on choice of data pack
parser.add_argument("--path_data", type=str, default="/notebook/Relations_Learning/Link_Prediction_Data/FB15K237/")
#parser.add_argument("--path_filters", type=str, default="/notebook/Relations_Learning/")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--opt_type", type=str, default='adam')

parser.add_argument('--dim', type = int, default = 200)
parser.add_argument('--l2', type = float, default = 0.0)
parser.add_argument('--scheduler_step', type=int, default=2, help="Scheduler step size")
parser.add_argument("--scheduler_gamma", type=float, default = 0.5, help="scheduler_gamma")
parser.add_argument('--momentum', type=float, default=0.9, help='momentum in Sgd')
parser.add_argument('--nesterov', type=bool, default=False, help='nesterov momentum in Sgd')
parser.add_argument(
    '--out_file', type=str,
    default='/notebook/Link-prediction/Warp_grid_search/output/result.txt',
    help='path tot output_file',
)

args = parser.parse_args()

print(args.n_epoch)
print(args.lr)
print(args.path_data)
print(args.batch_size)
print(args.opt_type)
print(args.dim)
print(args.l2)
print(args.scheduler_step)
print(args.scheduler_gamma)
print(args.momentum)
print(args.nesterov)
print(args.out_file)
print(os.path.dirname(os.path.abspath(__file__)))


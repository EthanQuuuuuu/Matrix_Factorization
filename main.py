import argparse
import numpy as np
from module import SynthesizeData, MF_Model, plot_error
from trainer import Trainer


def parse_agrs():
    parser = argparse.ArgumentParser()

    # Data synthesize settings
    parser.add_argument('--p', type=float, default=0.3, help='probability of non-zero item in L')
    parser.add_argument('--d', type=int, default=50, help='number of rows of matrix X')
    parser.add_argument('--n', type=int, default=50, help='number of rows of matrix X')
    parser.add_argument('--r', type=int, default=5, help='target rank of U and V')

    # Model settings
    parser.add_argument('--model_name', type=str, default='subgradient',choices=['subgradient', 'A_IRLS_combined','A_IRLS'], help='the algorithm to be used.')

    # Trainer related
    parser.add_argument('--max_iter_subG', type=int, default=2000, help='Iteration Num of Subgradient Method')
    parser.add_argument('--max_iter_A_ILRS_combined', type=int, default=30, help='Iteration Num of A_ILRS_combined')
    parser.add_argument('--max_iter_A_ILRS', type=int, default=100, help='Iteration Num of A_ILRS')
    parser.add_argument('--lr', type=int, default=3e-3, help='Learning rate of Subgradient Method')

    #Result related
    parser.add_argument('--save_dir',type=str, default='result/', help='result saving directory')

    args = parser.parse_args()
    return args


def main():
    # parse arguments
    args = parse_agrs()

    #Synthesize Data Loader
    X,L_star,U0,V0 = SynthesizeData(args)

    #Set the model
    model = MF_Model(args,X)

    #Begin Loop
    trainer = Trainer(model,args,X,L_star)

    #You can retrieve the final result U_star and V_star by replacing _ to a variable
    _, _ = trainer.train(args,U0,V0)

    #Plotting the error Curve unless having ground truth L_star
    plot_error(trainer.error_ls,args)


if __name__ == '__main__':
    main()

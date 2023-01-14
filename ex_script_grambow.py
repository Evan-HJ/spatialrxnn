from chemprop.args import TrainArgs
from chemprop.train.cross_validate import cross_validate
from chemprop.train.run_training import run_training

dir = 'cpf_cgrpre_100_grambow'

arguments = [
    '--data_path', 'datasets/b97d3.csv',
    '--dataset_type', 'regression',
    '--save_dir', dir,
    # '--checkpoint_dir', 'test_checkpoints_grambow',
    '--epochs', '100',
    '--reaction',
    '--explicit_h',
    '--split_type', 'scaffold_balanced',
    '--split_sizes', '0.9', '0.1', '0.0',
    '--aggregation', 'cgr-pretrain',
    # '--transformer_layers', '2',
    '--extra_metrics', 'mae'
]

args = TrainArgs().parse_args(arguments)
mean_score, std_score = cross_validate(args=args, train_func=run_training)

arguments = [
    '--data_path', 'datasets/wb97xd3.csv',
    '--dataset_type', 'regression',
    '--save_dir', dir,
    '--checkpoint_path', 'cpf_cgrpre_100_grambow/fold_0/model_0/model.pt',
    '--epochs', '100',
    '--reaction',
    '--explicit_h',
    '--split_type', 'scaffold_balanced',
    '--split_sizes', '0.8', '0.1', '0.1',
    '--aggregation', 'cgr-pretrain',
    # '--transformer_layers', '2',
    '--extra_metrics', 'mae'
]

args = TrainArgs().parse_args(arguments)
mean_score, std_score = cross_validate(args=args, train_func=run_training)

dir = 'cp_100_grambow'

arguments = [
    '--data_path', 'datasets/b97d3.csv',
    '--dataset_type', 'regression',
    '--save_dir', dir,
    '--checkpoint_path', 'cpf_cgrpre_100_grambow/fold_0/model_0/model.pt',
    # '--checkpoint_dir', dir,
    '--epochs', '100',
    # '--save_smiles_splits',
    # '--save_preds',
    '--reaction',
    '--explicit_h',
    '--split_type', 'scaffold_balanced',
    '--split_sizes', '0.9', '0.1', '0.0',
    '--aggregation', 'transformer',
    '--extra_metrics', 'mae',    
    '--transformer_layers', '2',
    '--freeze_mpn'
]

args = TrainArgs().parse_args(arguments)
mean_score, std_score = cross_validate(args=args, train_func=run_training)

arguments = [
    '--data_path', 'datasets/wb97xd3.csv',
    '--dataset_type', 'regression',
    '--save_dir', dir,
    '--checkpoint_path', 'cp_100_grambow/fold_0/model_0/model.pt',
    # '--checkpoint_dir', dir,
    '--epochs', '100',
    '--save_smiles_splits',
    '--save_preds',
    '--reaction',
    '--explicit_h',
    '--split_type', 'scaffold_balanced',
    '--split_sizes', '0.8', '0.1', '0.1',
    '--aggregation', 'transformer',
    '--extra_metrics', 'mae',    
    '--transformer_layers', '2',
    '--freeze_mpn'
]

args = TrainArgs().parse_args(arguments)
mean_score, std_score = cross_validate(args=args, train_func=run_training)

from argparse import Namespace
from typing import List, Union, Tuple
import pickle, os

import numpy as np
from rdkit import Chem
import torch
import torch.nn as nn

from .mpn import MPN
from chemprop.args import TrainArgs
from chemprop.features import BatchMolGraph
from chemprop.nn_utils import get_activation_function, initialize_weights, pad_tensor, get_mask, pad_attn_bias
from chemprop.models.transformer import Encoder, EncoderLayer, GaussianLayer

class Model(nn.Module):
    """Base class for molecule and reaction models, which only differ in their encoding."""

    def __init__(self, args: TrainArgs):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        """
        super(Model, self).__init__()

        self.classification = args.dataset_type == 'classification'
        self.multiclass = args.dataset_type == 'multiclass'
        self.loss_function = args.loss_function

        if hasattr(args, 'train_class_sizes'):
            self.train_class_sizes = args.train_class_sizes
        else:
            self.train_class_sizes = None

        # when using cross entropy losses, no sigmoid or softmax during training. But they are needed for mcc loss.
        if self.classification or self.multiclass:
            self.no_training_normalization = args.loss_function in ['cross_entropy', 'binary_cross_entropy']

        self.output_size = args.num_tasks
        if self.multiclass:
            self.output_size *= args.multiclass_num_classes
        if self.loss_function == 'mve':
            self.output_size *= 2  # return means and variances
        if self.loss_function == 'dirichlet' and self.classification:
            self.output_size *= 2  # return dirichlet parameters for positive and negative class
        if self.loss_function == 'evidential':
            self.output_size *= 4  # return four evidential parameters: gamma, lambda, alpha, beta

        if self.classification:
            self.sigmoid = nn.Sigmoid()

        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)

        if self.loss_function in ['mve', 'evidential', 'dirichlet']:
            self.softplus = nn.Softplus()

    def create_encoder(self, args: TrainArgs):
        """
        Creates the encoder for the model.

        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    def create_ffn(self, args: TrainArgs) -> None:
        """
        Creates the feed-forward layers for the model.

        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        """
        self.multiclass = args.dataset_type == 'multiclass'
        if self.multiclass:
            self.num_classes = args.multiclass_num_classes
        if args.features_only:
            first_linear_dim = args.features_size
        else:
            if args.reaction_solvent:
                first_linear_dim = args.hidden_size + args.hidden_size_solvent
            else:
                first_linear_dim = args.hidden_size * args.number_of_molecules
            if args.use_input_features:
                first_linear_dim += args.features_size

        if args.atom_descriptors == 'descriptor':
            first_linear_dim += args.atom_descriptors_size
        
        # if args.use_spatial_features:
        #     first_linear_dim += args.num_kernels

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)

        # Create FFN layers
        if args.ffn_num_layers == 1:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, self.output_size)
            ]
        else:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.ffn_hidden_size)
            ]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
                ])
            ffn.extend([
                activation,
                dropout,
                nn.Linear(args.ffn_hidden_size, self.output_size),
            ])

        #@evan
        # # If spectra model, also include spectra activation
        # if args.dataset_type == 'spectra':
        #     if args.spectra_activation == 'softplus':
        #         spectra_activation = nn.Softplus()
        #     else:  # default exponential activation which must be made into a custom nn module
        #         class nn_exp(torch.nn.Module):
        #             def __init__(self):
        #                 super(nn_exp, self).__init__()

        #             def forward(self, x):
        #                 return torch.exp(x)

        #         spectra_activation = nn_exp()
        #     ffn.append(spectra_activation)

        # Create FFN model
        self.ffn = nn.Sequential(*ffn)

        # if args.checkpoint_frzn is not None:
        #     if args.frzn_ffn_layers > 0:
        #         for param in list(self.ffn.parameters())[0:2 * args.frzn_ffn_layers]:  # Freeze weights and bias for given number of layers
        #             param.requires_grad = False

    def create_gaussian_layer(self, args):
        self.gbf = GaussianLayer(args)

    def forward(self, *input):
        """
        Defines the computation performed at every call.

        Should be overridden by all subclasses.
        """
        raise NotImplementedError

class MoleculeModel(Model):
    """A :class:`MoleculeModel` is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self, args: TrainArgs):
        super(MoleculeModel, self).__init__(args)

    def create_encoder(self, args: TrainArgs) -> None:
        """
        Creates the message passing encoder for the model.

        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        """
        self.encoder = MPN(args)

        # if args.freeze_mpn:
        #     for param in self.encoder.parameters():
        #         param.requires_grad = False
        
        #@evan
        # if args.checkpoint_frzn is not None:
        #     if args.freeze_first_only:  # Freeze only the first encoder
        #         for param in list(self.encoder.encoder.children())[0].parameters():
        #             param.requires_grad = False
        #     else:  # Freeze all encoders
        #         for param in self.encoder.parameters():
        #             param.requires_grad = False
        
    def fingerprint(self,
                    batch: Union[List[List[str]], List[List[Chem.Mol]], List[List[Tuple[Chem.Mol, Chem.Mol]]], List[BatchMolGraph]],
                    features_batch: List[np.ndarray] = None,
                    atom_descriptors_batch: List[np.ndarray] = None,
                    atom_features_batch: List[np.ndarray] = None,
                    bond_features_batch: List[np.ndarray] = None,
                    fingerprint_type: str = 'MPN') -> torch.Tensor:
        """
        Encodes the latent representations of the input molecules from intermediate stages of the model.

        :param batch: A list of list of SMILES, a list of list of RDKit molecules, or a
                      list of :class:`~chemprop.features.featurization.BatchMolGraph`.
                      The outer list or BatchMolGraph is of length :code:`num_molecules` (number of datapoints in batch),
                      the inner list is of length :code:`number_of_molecules` (number of molecules per datapoint).
        :param features_batch: A list of numpy arrays containing additional features.
        :param atom_descriptors_batch: A list of numpy arrays containing additional atom descriptors.
        :param fingerprint_type: The choice of which type of latent representation to return as the molecular fingerprint. Currently
                                 supported MPN for the output of the MPNN portion of the model or last_FFN for the input to the final readout layer.
        :return: The latent fingerprint vectors.
        """
        if fingerprint_type == 'MPN':
            return self.encoder(batch, features_batch, atom_descriptors_batch,
                                atom_features_batch, bond_features_batch)
        elif fingerprint_type == 'last_FFN':
            return self.ffn[:-1](self.encoder(batch, features_batch, atom_descriptors_batch,
                                              atom_features_batch, bond_features_batch))
        else:
            raise ValueError(f'Unsupported fingerprint type {fingerprint_type}.')

    #@evan
    def freeze_encoders(self, args: TrainArgs):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self,
                batch: Union[List[List[str]], List[List[Chem.Mol]], List[List[Tuple[Chem.Mol, Chem.Mol]]], List[BatchMolGraph]],
                features_batch: List[np.ndarray] = None,
                atom_descriptors_batch: List[np.ndarray] = None,
                atom_features_batch: List[np.ndarray] = None,
                bond_features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Runs the :class:`MoleculeModel` on input.

        :param batch: A list of list of SMILES, a list of list of RDKit molecules, or a
                      list of :class:`~chemprop.features.featurization.BatchMolGraph`.
                      The outer list or BatchMolGraph is of length :code:`num_molecules` (number of datapoints in batch),
                      the inner list is of length :code:`number_of_molecules` (number of molecules per datapoint).
        :param features_batch: A list of numpy arrays containing additional features.
        :param atom_descriptors_batch: A list of numpy arrays containing additional atom descriptors.
        :param atom_features_batch: A list of numpy arrays containing additional atom features.
        :param bond_features_batch: A list of numpy arrays containing additional bond features.
        :return: The output of the :class:`MoleculeModel`, containing a list of property predictions
        """
        #@evan: batch passes in as BatchMolGraph object

        #@evan: Encode -> atom features -> FFN
        output = self.ffn(self.encoder(batch, features_batch, atom_descriptors_batch,
                                       atom_features_batch, bond_features_batch))

        if self.classification and not (self.training and self.no_training_normalization) and self.loss_function != 'dirichlet':
            output = self.sigmoid(output)
        if self.multiclass:
            output = output.reshape((output.shape[0], -1, self.num_classes))  # batch size x num targets x num classes per target
            if not (self.training and self.no_training_normalization) and self.loss_function != 'dirichlet':
                output = self.multiclass_softmax(output)  # to get probabilities during evaluation, but not during training when using CrossEntropyLoss

        #@evan
        # Modify multi-input loss functions
        # if self.loss_function == 'mve':
        #     means, variances = torch.split(output, output.shape[1] // 2, dim=1)
        #     variances = self.softplus(variances)
        #     output = torch.cat([means, variances], axis=1)
        # if self.loss_function == 'evidential':
        #     means, lambdas, alphas, betas = torch.split(output, output.shape[1]//4, dim=1)
        #     lambdas = self.softplus(lambdas)  # + min_val
        #     alphas = self.softplus(alphas) + 1  # + min_val # add 1 for numerical contraints of Gamma function
        #     betas = self.softplus(betas)  # + min_val
        #     output = torch.cat([means, lambdas, alphas, betas], dim=1)
        # if self.loss_function == 'dirichlet':
        #     output = nn.functional.softplus(output) + 1

        return output

class ReactionModel2(Model):
    '''
    We do things here
    '''
    def __init__(self, args: TrainArgs):
        super(ReactionModel2, self).__init__(args)

        self.max_num_atoms = args.max_num_atoms
        self.use_spatial_features = args.use_spatial_features
        self.hidden_size = args.hidden_size

        self.device = args.device

        self.lin = nn.Linear(args.num_kernels, args.hidden_size)
        self.w1 = nn.Parameter(torch.zeros(1))

    def create_encoder(self, args: TrainArgs):
        self.m1 = MoleculeModel(args)
        self.m1.create_encoder(args)

        # initialize_weights(self.m1)
        return
    
    def freeze_encoders(self, args: TrainArgs):
        for param in self.m1.parameters():
            param.requires_grad = False

    def forward(self,
                batch: Union[List[List[str]], List[List[Chem.Mol]], List[List[Tuple[Chem.Mol, Chem.Mol]]], List[BatchMolGraph]],
                features_batch: List[np.ndarray] = None,
                atom_descriptors_batch: List[np.ndarray] = None,
                atom_features_batch: List[np.ndarray] = None,
                bond_features_batch: List[np.ndarray] = None) -> torch.FloatTensor:

        embeddings = self.m1.fingerprint(batch)

        mol_vecs = []
        for mol_vec in embeddings:
            # mol_vec = mol_vec.sum(dim=0) / mol_vec.shape[0]
            mol_vec = mol_vec.mean(dim=0)
            mol_vecs.append(mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0)  # (num_molecules, hidden_size)

        if self.use_spatial_features:
            #Attention features
            mol_graph = batch[0]
            edge_type, edge_delta_dist = mol_graph.get_attn_features()

            embeddings = [torch.cat([torch.zeros((1,self.hidden_size)).to(self.device), x], 0) for x in embeddings]
            embeddings_mask = get_mask(embeddings, self.max_num_atoms).to(self.device) #8x1x30
            
            edge_type_tensor = pad_attn_bias(edge_type, self.max_num_atoms).to(self.device) #8x30x30
            edge_dist_tensor = pad_attn_bias(edge_delta_dist, self.max_num_atoms).to(self.device) #8x30x30
            
            spatial_features = self.gbf(edge_dist_tensor, edge_type_tensor) #8x30x30x60

            if embeddings_mask is not None: #Mask: 8x1x30x1, 8x30x1x1 
                spatial_features = spatial_features.masked_fill(embeddings_mask.unsqueeze(1).transpose(-1,-2) == 0, 0)
                spatial_features = spatial_features.masked_fill(embeddings_mask.unsqueeze(1).transpose(1, 3) == 0, 0)

            spatial_features[:,0,:,:] = 0
            spatial_features[:,:,0,:] = 0

            sum_of_distances = spatial_features.sum(dim=2) #8x30x60
            aggr_distance_features = sum_of_distances.sum(dim=1) / sum_of_distances.shape[1] # batch*atoms*kernels (8x30x60) => batch*kernels (8x60)

            aggr_distance_features = self.lin(aggr_distance_features) #8x300

            mol_vecs = mol_vecs + aggr_distance_features * self.w1
 
        preds = self.ffn(mol_vecs)

        return preds

class ReactionModel(Model):
    '''
    We do things here
    '''
    def __init__(self, args: TrainArgs):
        super(ReactionModel, self).__init__(args)

        self.max_num_atoms = args.max_num_atoms
        self.device = args.device
        self.print_attention = args.print_attention

        # self.cls = nn.Parameter(torch.ones(1, args.hidden_size))
        self.lin = nn.Linear(args.num_kernels, args.hidden_size)
        # self.w0 = nn.Parameter(torch.ones(1))
        self.w1 = nn.Parameter(torch.zeros(1))

        self.w2 = nn.Parameter(torch.ones(1))
        self.w3 = nn.Parameter(torch.zeros(1))

    def create_encoder(self, args: TrainArgs):
        self.m1 = MoleculeModel(args)
        self.m1.create_encoder(args)

        # initialize_weights(self.m1)
        return
    
    def create_transformer(self, args: TrainArgs):
        self.transformer = Encoder(EncoderLayer(args), args)
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def freeze_encoders(self, args: TrainArgs):
        for param in self.m1.parameters():
            param.requires_grad = False

        # for param in self.transformer.parameters():
        #     param.requires_grad = False

    def save_smiles_attn(self, smiles):
        if os.path.exists('attention_smiles.pkl'):
            with open('attention_smiles.pkl', 'rb') as f:
                smiles_list = pickle.load(f)
        else:
            smiles_list = []

        smiles_list.extend(smiles)

        with open('attention_smiles.pkl', 'wb') as f:
            pickle.dump(smiles_list, f)

    def forward(self,
                batch: Union[List[List[str]], List[List[Chem.Mol]], List[List[Tuple[Chem.Mol, Chem.Mol]]], List[BatchMolGraph]],
                features_batch: List[np.ndarray] = None,
                atom_descriptors_batch: List[np.ndarray] = None,
                atom_features_batch: List[np.ndarray] = None,
                bond_features_batch: List[np.ndarray] = None) -> torch.FloatTensor:

        embeddings = self.m1.fingerprint(batch)
        # embeddings = torch.stack(embeddings, dim = 0)

        #Mean aggregation 
        mol_vecs = []
        for mol_vec in embeddings:
            mol_vec = mol_vec.sum(dim=0) / mol_vec.shape[0]
            mol_vecs.append(mol_vec)
        mol_vecs = torch.stack(mol_vecs, dim=0)  # (batch, hidden_size) 8x300

        #Prepare embeddings to be passed into transformer
        embeddings_with_cls = []
        for indx in range(len(embeddings)):
            embeddings_with_cls.append(torch.cat([mol_vecs[indx].unsqueeze(0), embeddings[indx]], 0))

        embeddings = embeddings_with_cls
        # embeddings = [torch.cat([self.cls, x], 0) for x in embeddings]
        # embeddings = torch.cat([mol_vecs.unsqueeze(1), embeddings], 1)

        embeddings_mask = get_mask(embeddings, self.max_num_atoms).to(self.device) #8x1x30
        embeddings = pad_tensor(embeddings, self.max_num_atoms).to(self.device) #8x30x300

        #Attention features
        mol_graph = batch[0]
        
        if (self.print_attention and self.training == False):
            self.save_smiles_attn(mol_graph.get_smiles())

        edge_type, edge_delta_dist = mol_graph.get_attn_features()

        edge_type_tensor = pad_attn_bias(edge_type, self.max_num_atoms).to(self.device) #8x30x30
        edge_dist_tensor = pad_attn_bias(edge_delta_dist, self.max_num_atoms).to(self.device) #8x30x30
        
        spatial_features = self.gbf(edge_dist_tensor, edge_type_tensor) #8x30x30x60

        if embeddings_mask is not None: #Mask: 8x1x30x1, 8x30x1x1 
            spatial_features = spatial_features.masked_fill(embeddings_mask.unsqueeze(1).transpose(-1,-2) == 0, 0)
            spatial_features = spatial_features.masked_fill(embeddings_mask.unsqueeze(1).transpose(1, 3) == 0, 0)

        spatial_features[:,0,:,:] = 0
        spatial_features[:,:,0,:] = 0

        sum_of_distances = spatial_features.sum(dim=2) #8x30x60

        sum_of_distances = self.lin(sum_of_distances) #8x30x300

        embeddings = embeddings + sum_of_distances * self.w1

        #Pass into transformer
        embeddings = self.transformer(embeddings, embeddings_mask, spatial_features) #8x30x300
        embeddings = embeddings[:, 0, :] #Take CLS: 8x300

        #Add spatial features to GNN outputs.
        # aggr_distance_features = sum_of_distances.sum(dim=1) / sum_of_distances.shape[1] # batch*atoms*kernels (8x30x60) => batch*kernels (8x60)
        # mol_vecs = torch.cat([mol_vecs, aggr_distance_features], dim = 1) #8x360
        
        #Pass into FFN
        final_embeddings = torch.mul(mol_vecs, self.w2) + torch.mul(embeddings, self.w3)
        # final_embeddings = mol_vecs + embeddings
        # final_embeddings = embeddings
        preds = self.ffn(final_embeddings)

        return preds#, mae

def build_model(args) -> nn.Module:
    # if args.reaction: 
    # else:
    # model = MoleculeModel(args)
    
    if args.aggregation == "cgr-pretrain":
        model = ReactionModel2(args)
        if args.use_spatial_features:
            model.create_gaussian_layer(args)
    elif args.aggregation == "transformer-finetune": #Different arg because we want to save results to a different file later
        model = ReactionModel(args)
        model.create_transformer(args)
        model.create_gaussian_layer(args)
    elif args.aggregation == "transformer":
        model = ReactionModel(args)
        model.create_transformer(args)
        model.create_gaussian_layer(args)
    else:
        model = MoleculeModel(args)

    model.create_encoder(args)
    model.create_ffn(args)

    # initialize_weights(model)
    
    return model
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
import torch_geometric as PyG
import torch.nn.functional as F

class MPL_1(MessagePassing): #Message passing layer
    def __init__(self, input_dimension=None, output_dimension=None, activation=torch.relu, aggregation="add"):
        super().__init__(aggr=aggregation)
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.lin = nn.Linear(self.input_dimension, self.output_dimension)
        self.act = activation
        
    def forward(self, edge_index, node_attr, edge_attr):
        return self.propagate(edge_index, x=node_attr, edge_attr=edge_attr)
    
    def message(self, x_i, x_j, edge_attr):
        attr = torch.cat([x_i,x_j,edge_attr],1) #in_features is the size of this object (n_atoms, in_features)
        # print(x_i)
        # print(attr)
        attr = self.lin(attr)
        attr = self.act(attr)

        return attr
        
    def update(self, aggregated): #Here we return the new tensor that is the new node embeddings. Can use x_i x_j and so on
#         print(x_j)
        return aggregated

class MPL_2(MessagePassing):
    """
    Message is [x_i,x_j,edge_attr]
    2 layer transforation
    """
    def __init__(self, input_dimension=None, hidden_dimension=None, output_dimension=None, activation=torch.relu, aggregation="add"):
        super().__init__(aggr=aggregation)
        self.input_dimension = input_dimension
        self.hidden_dimension = hidden_dimension
        self.output_dimension = output_dimension
        self.lin_1 = nn.Linear(self.input_dimension, self.hidden_dimension)
        self.lin_2 = nn.Linear(self.hidden_dimension, self.output_dimension)
        self.act = activation
        
    def forward(self, edge_index, node_attr, edge_attr):
        return self.propagate(edge_index, x=node_attr, edge_attr=edge_attr)
    
    def message(self, x_i, x_j, edge_attr):
        attr = torch.cat([x_i,x_j,edge_attr],1) #in_features is the size of this object (n_atoms, in_features)
        attr = self.lin_1(attr)
        attr = self.act(attr)
        attr = self.lin_2(attr)
        attr = self.act(attr)
        return attr
        
    def update(self, aggregated): #Here we return the new tensor that is the new node embeddings. Can use x_i x_j and so on
        return aggregated

class MPL_3(MessagePassing):
    """
    message consists of x_j and edge_attr
    2 linear layer + activation
    update: [x_i, aggregated] 
    """
    def __init__(self, input_dimension=None, hidden_dimension=None, output_dimension=None, activation=F.relu, aggregation="add"):
        super().__init__(aggr=aggregation)
        self.input_dimension = input_dimension
        self.hidden_dimension = hidden_dimension
        self.output_dimension = output_dimension
        self.lin_1 = nn.Linear(self.input_dimension, self.hidden_dimension)
        self.lin_2 = nn.Linear(self.hidden_dimension, self.output_dimension)
        self.act = activation
        
    def forward(self, edge_index, node_attr, edge_attr):
        return self.propagate(edge_index, x=node_attr, edge_attr=edge_attr)
    
    def message(self, x_j, edge_attr):
        attr = torch.cat([x_j,edge_attr],1) #in_features is the size of this object (n_atoms, in_features)
        attr = self.act(self.lin_1(attr))
        attr = self.act(self.lin_2(attr))
        return attr
        
    def update(self, aggregated):
        return aggregated





# molecules = []

# molecules.append(
#     {
#         "edge_index":torch.Tensor([[0,1],[1,0]]),
#         "edge_attr":torch.Tensor([[0,1,0],[0,2,0]]),
#         "node_attr":torch.Tensor([[1,0],[0,1]]),
#         "graph_attr":torch.Tensor([-1.0, -2.0]),
#         "y":torch.Tensor([-3])
#     }
# )

# def create_data(mol_dict):
#     data = Data()
#     data.node_attr = mol_dict["node_attr"]
#     data.num_nodes = data.node_attr.shape[0]
#     data.edge_attr = mol_dict["edge_attr"]
#     data.edge_index = mol_dict["edge_index"].to(dtype=torch.long)
#     data.y = mol_dict["y"]
#     data.graph_attr = mol_dict["graph_attr"]
#     return data

# data = create_data(molecules[0])

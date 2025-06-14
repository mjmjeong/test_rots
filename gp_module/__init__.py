import torch
from torch.utils.data import TensorDataset, DataLoader

from gp_module.model import *
from gp_module.kernel import *
from gp_module.likelihood import *
from gp_module.loss import *
from gp_module.uncertainty import *

class Motion_GP():
    def __init__(self, args):
        self.args = args
        self.device = args.device
        
        self.transls_gp = eval(self.args.gp.transls_model)
        self.rots_gp = eval(self.args.gp.rots_model)

    def get_dataset(self, transls, rots,  canonical_idx=0, confidence=None):
        self.set_bbox(transls)
        transls = self.bbox_normalize(transls)

        num_data_points = transls.shape[0]
        total_time = transls.shape[1] # Time 
        can_xyz = transls[:,canonical_idx]
        can_xyz = can_xyz.unsqueeze(1).repeat(1, total_time, 1)
        tgt_time = torch.linspace(0, 1, total_time)
        tgt_time = tgt_time.view(1, total_time, 1).repeat(num_data_points, 1, 1)
        train_x = torch.cat((can_xyz, tgt_time), dim=-1)
        train_y = torch.cat((transls, rots, confidence), -1)
    
        train_x = train_x.reshape(num_data_points*total_time, -1)
        train_y = train_y.reshape(num_data_points*total_time, -1)

#        if self.args.gp.confidence_thred is not None: 
#            confidence_mask = train_y[:,-1] > self.args.gp.confidence_thred
#            train_x = train_x[confidence_mask]
#            train_y = train_y[confidence_mask]
            
        return train_x[:10000], train_y[:10000] # TODO
        
    def init_gp(self, train_x, train_y):

        train_y_transls = train_y[:,:3]
        train_y_rots = train_y[:,3:-1]

        # likelihood        
        self.transls_likelihood = self.transls_gp.get_likelihood(self.args, "transls")
        self.rots_likelihood = self.transls_gp.get_likelihood(self.args, "rots")

        # gp_model       
        self.transls_gp_model = self.transls_gp.init_from_data(self.args, 'transls', train_x, train_y_transls, self.transls_likelihood)
        self.rots_gp_model = self.rots_gp.init_from_data(self.args, 'rots', train_x, train_y_rots, self.rots_likelihood)

        # mll (loss)
        self.transls_mll = self.transls_gp.get_mll(self.transls_likelihood, self.transls_gp_model)
        self.rots_mll = self.rots_gp.get_mll(self.rots_likelihood, self.rots_gp_model)

        # optimizer
        self.transls_optimizer = self.transls_gp.get_optimizer(self.args, 'transls', self.transls_likelihood, self.transls_gp_model)
        self.rots_optimizer = self.rots_gp.get_optimizer(self.args, 'rots', self.rots_likelihood, self.rots_gp_model)   
    
    def set_bbox(self, data):
        min_vals = data.min(dim=0, keepdim=True).values.min(dim=1, keepdim=True).values.squeeze(0)
        max_vals = data.max(dim=0, keepdim=True).values.max(dim=1, keepdim=True).values.squeeze(0)

        min_for_scaling = torch.floor(min_vals)
        max_for_scaling = torch.ceil(max_vals)

        self.bbox = {'min': min_for_scaling, 'max': max_for_scaling}
    
    def bbox_normalize(self, data):
        min_for_scaling = self.bbox['min']
        max_for_scaling = self.bbox['max']

        denominator = (max_for_scaling - min_for_scaling).unsqueeze(0)
        min_for_scaling_expanded = min_for_scaling.unsqueeze(0)

        data_scaled = 2 * (data - min_for_scaling_expanded) / denominator - 1
        return data_scaled
    
    def remove_gp(self):
        try:
            del self.transls_gp_model
            del self.rots_gp_model
            del self.transls_likelihood
            del self.rots_likelihood
            
            del self.transls_optimizer
            del self.rots_optimizer
        except:
            pass
    
    def set_device(self):
        self.transls_gp_model.to(self.device)
        self.transls_likelihood.to(self.device)
        self.rots_gp_model.to(self.device)
        self.rots_likelihood.to(self.device) 

    def set_mode(self, type_):
        if type_ == 'train':
            self.transls_gp_model.train()
            self.transls_likelihood.train()
            self.rots_gp_model.train()
            self.rots_likelihood.train()       

        elif type_ == 'eval':
            self.transls_gp_model.eval()
            self.transls_likelihood.eval()
            self.rots_gp_model.eval()
            self.rots_likelihood.eval()       

    def fitting_gp(self, train_x, train_y):
        if self.args.gp.transls_model in ['MultitaskGPModel']: 
            self.fitting_exact_gp(train_x, train_y) # full
        elif self.args.gp.transls_model in []:
            self.fitting_variational_gp(train_x, train_y)
            raise NotImplementedError("Check training loop selection")

    def fitting_exact_gp(self, train_x, train_y):
        # Initialization
        self.remove_gp()
        self.init_gp(train_x, train_y)
        
        self.set_device()
        self.set_mode('train')

        train_x = train_x.to(self.device)
        train_y = train_y.to(self.device)

        y_transls = train_y[:, :3]
        y_rots = train_y[:, 3:-1] 
        y_confidence = train_y[:, -1:]  

        for i in range(self.args.gp.epochs):
            # reset gradient
            self.transls_optimizer.zero_grad()
            self.rots_optimizer.zero_grad()
            with gpytorch.settings.use_toeplitz(False), gpytorch.settings.max_root_decomposition_size(30):
                # forward
                print("train_x:", train_x.shape)
                transls_output = self.transls_gp_model(train_x)
                transls_loss = self.transls_mll(transls_output, y_transls)
                #, batch_confidence) 

                rots_output = self.rots_gp_model(train_x)
                rots_loss = self.rots_mll(rots_output, y_rots)
                #, batch_confidence)

                loss = transls_loss + rots_loss

                # backward
                loss.backward()
                self.transls_optimizer.step()
                self.rots_optimizer.step()

            if (i + 1) % 1 == 0:
                print(f"[{i+1}] Loss: {loss.item():.4f}")

    def fitting_variational_gp(self, transls, rots, canonical_idx=0, confidence=None):
        # Initialization
        self.remove_gp()
        self.init_gp(transls, rots, canonical_idx)
        
        num_data_points = transls.shape[0]
        total_time = transls.shape[1] # Time 
        can_xyz = transls[:,canonical_idx]
        can_xyz = can_xyz.unsqueeze(1).repeat(1, total_time, 1)
        tgt_time = torch.linspace(0, 1, total_time)
        tgt_time = tgt_time.view(1, total_time, 1).repeat(num_data_points, 1, 1)
        train_x = torch.cat((can_xyz, tgt_time), dim=-1)
        train_y = torch.cat((transls, rots, confidence), -1)
    
        train_x = train_x.reshape(num_data_points*total_time, -1)
        train_y = train_y.reshape(num_data_points*total_time, -1)

        if self.args.gp.confidence_thred is not None: 
            confidence_mask = train_y[:,-1] > self.args.gp.confidence_thred
            train_x = train_x[confidence_mask]
            train_y = train_y[confidence_mask]
            
        self.set_device()
        self.set_mode('train')

        # define dataloader
        train_dataset = TensorDataset(train_x, train_y)
        train_loader = DataLoader(train_dataset, batch_size=self.args.gp.batch_size, shuffle=True) 
        
        for i in range(self.args.gp.epochs):
            for batch_idx, (batch_x, batch_y_full) in enumerate(train_loader):
                batch_x = batch_x.to(self.device)
                batch_y_full = batch_y_full.to(self.device)

                batch_y_transls = batch_y_full[:, :3]
                batch_y_rots = batch_y_full[:, 3:-1] 
                batch_confidence = batch_y_full[:, -1:]  

                # reset gradient
                self.transls_optimizer.zero_grad()
                self.rots_optimizer.zero_grad()
 
                # forward
                transls_output = self.transls_gp_model(batch_x)
                transls_loss = self.transls_mll(transls_output, batch_y_transls)
                #, batch_confidence) 
                # TODO: confidence aware MLL
                
                rots_output = self.rots_gp_model(batch_x)
                rots_loss = self.rots_mll(rots_output, batch_y_rots)
                #, batch_confidence)

                loss = transls_loss + rots_loss

                # backward
                loss.backward()
                self.transls_optimizer.step()
                self.rots_optimizer.step()

                if (i + 1) % 10 == 0:
                    print(f"[{i+1}] Loss: {loss.item():.4f}")

    def sampling(self, data, keep=False):
        self.set_mode('eval')
        x = data
        transls_output = self.transls_gp_model(x)
        rots_output = self.rots_gp_model(x)

    def recon_loss(self, transls_curr, rots_curr, confidence):
        pass

    # utils
    def plot_traj(self, data, name=None):
        if data.dim(-1) == 3:
            dim_labels = ['X', 'Y', 'Z']  # Labels for dimensions
        elif data.dim(-1) == 4:
            dim_labels = ['quat_0', 'quat_1', 'quat_2', 'quat_3']  # Labels for dimensions
    
        # Create a single figure
        plt.figure(figsize=(12, 12))

        for dim in range(3):
            plt.subplot(3, 1, dim + 1)  # 3 rows, 1 column
            for i in range(min(40, transls.size(0))):  # Limit to 40 samples or fewer
                plt.plot(transls[i, :, dim].detach().cpu().numpy(), label=f'Sample {i}')
            plt.title(f'Translation - {dim_labels[dim]} axis')
            plt.xlabel('Time Step')
            plt.ylabel(f'{dim_labels[dim]} Value')
            plt.grid(True)

        # Save plot
        plt.tight_layout()
        if name is None:
            name = 'translation_all_axes.png'
        else:
            name = f'{name}.png'
        plt.savefig(name, dpi=300)
        plt.close()
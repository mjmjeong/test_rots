import torch
import time
import numpy as np
import cupy as cp
from cuml import HDBSCAN, KMeans
from flow3d.vis.utils import draw_keypoints_video, get_server, project_2d_tracks
from viser import ViserServer
from matplotlib.pyplot import get_cmap
from loguru import logger as guru
import gc

def vis_tracks_3d(
    server: ViserServer,
    vis_tracks: np.ndarray,
    vis_label: np.ndarray | None = None,
    name: str = "tracks",
    colors: np.ndarray | None = None,
):
    """
    :param vis_tracks (np.ndarray): (N, T, 3)
    :param vis_label (np.ndarray): (N)
    """
    cmap = get_cmap("gist_rainbow")
    if vis_label is None:
        vis_label = np.linspace(0, 1, len(vis_tracks))
    if colors is None:
        colors = cmap(np.asarray(vis_label))[:, :3]
    guru.info(f"{colors.shape=}, {vis_tracks.shape=}")
    N, T = vis_tracks.shape[:2]
    vis_tracks = np.asarray(vis_tracks)
    for i in range(N):
        server.scene.add_spline_catmull_rom(
            f"/{name}/{i}/spline", vis_tracks[i], color=colors[i], segments=T - 1
        )
        server.scene.add_point_cloud(
            f"/{name}/{i}/start",
            vis_tracks[i, [0]],
            colors=colors[i : i + 1],
            point_size=0.01,
            point_shape="circle",
        )
        server.scene.add_point_cloud(
            f"/{name}/{i}/end",
            vis_tracks[i, [-1]],
            colors=colors[i : i + 1],
            point_size=0.01,
            point_shape="diamond",
        )


###############################################################
# Rsample 
###############################################################
def estimate_std(x, x_conf, iteration, args):
    type_ = args.input_rsample
    if type_ == 'lookup':
        pass

    elif type_ == 'softplus':
        pass

    elif type_ == 'fix':
        return args.rsample_std

    elif type_ == 'annealing':
        # iteration: global iteration
        # iteration: (curretn local iteration)
        pass

    else:
        raise NotImplementedError("Please check gp.x_rsample type!")

###############################################################
# Inducing points
###############################################################
def create_grid_inducing_points(x_range, nx, y_range, ny, z_range, nz, 
                               t_range=None, nt=None):
    """
    Create inducing points on a regular grid for spatiotemporal data
    
    Args:
        x_range, y_range, z_range, t_range: tuples of (min, max) for each dimension
        nx, ny, nz, nt: number of grid points in each dimension
    """
    
    if t_range is not None: 
        x_grid = torch.linspace(x_range[0], x_range[1], nx)
        y_grid = torch.linspace(y_range[0], y_range[1], ny)
        z_grid = torch.linspace(z_range[0], z_range[1], nz)
        t_grid = torch.linspace(t_range[0], t_range[1], nt)
        X, Y, Z, T = torch.meshgrid(x_grid, y_grid, z_grid, t_grid, indexing='ij')
            # Flatten and stack
        inducing_points = torch.stack([
            X.flatten(), Y.flatten(), Z.flatten(), T.flatten()
        ], dim=1)
        

    else:
        x_grid = torch.linspace(x_range[0], x_range[1], nx)
        y_grid = torch.linspace(y_range[0], y_range[1], ny)
        z_grid = torch.linspace(z_range[0], z_range[1], nz)
        X, Y, Z = torch.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
    
        inducing_points = torch.stack([
            X.flatten(), Y.flatten(), Z.flatten()
        ], dim=1)
    
    return inducing_points


def create_adaptive_inducing_points(train_x, others, args):
    num_inducing=args.gp.inducing_num
    method=args.gp.inducing_method
    inducing_min=args.gp.inducing_min
    inducing_max=args.gp.inducing_max
    add_noise_scale=args.gp.inducing_point_noise_scale

    nx = args.gp.nx
    nt = args.gp.nt
    
    # full traj
    conf = others['conf'] # just for visualization
    traj = others['traj'] # normed trajectory

    xyz_interp = torch.tensor(others['traj']).cpu()
    T = xyz_interp.size(1)
    N = num_tracks = xyz_interp.size(0)
    time_gap = T // (nt-1)

    if 'kmeans' in method or 'hdbscan' in method:
        assert (method in ['vel_kmeans', 'vel_hdbscan',
                            'chronos_kmeans', 'chronos_hdbscan',
                            'vel_chronos_kmeans', 'vel_chronos_hdbscan',
                            ])
        
        mode = method.split('_')[-1]

        if 'vel' in method:
            vel = xyz_interp[:,1:,:] - xyz_interp[:,:-1,:]
        else:
            vel = xyz_interp

        if 'chronos' in method:
            with torch.no_grad():
                from flow3d.init_utils import get_chronos_embeddings
                embeddings = get_chronos_embeddings(vel, concat=True)  
                embeddings = embeddings[:,0, :] # first idx = global context
        else:
            embeddings = vel.reshape(xyz_interp.size(0), -1)

        torch.cuda.empty_cache()
        gc.collect()
            
        vel_dirs = cp.asarray(embeddings.to(dtype=torch.float32))

        if mode == "kmeans":
            model = KMeans(n_clusters=nx)
        elif mode == 'hdbscan':
            model = HDBSCAN(min_cluster_size=20, max_cluster_size=num_tracks // 4)

        model.fit(vel_dirs)
        labels = model.labels_
        num_bases = labels.max().item() + 1

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        
        centers = model.cluster_centers_
        centers_expanded = torch.tensor(centers).unsqueeze(1).to(device)
        embeddings_expanded = embeddings.unsqueeze(0).to(device)

        distances = torch.sum((centers_expanded - embeddings_expanded) ** 2, dim=2)  # (basis, traj_num)
        closest_indices = torch.argmin(distances, dim=1).cpu()

        """
        server = get_server(port=8890)
        labels = np.linspace(0, 1, nx)
        cmap = get_cmap("gist_rainbow")
        colors = cmap(np.asarray(labels))[:, :3]

        for i in range(nx):
            mask = np.array(torch.tensor(model.labels_== i).cpu())
#            num_vis = mask.sum().item()
            num_vis = 8
            xyz_np = np.array(xyz_interp.cpu())
            colors_cluster = colors[i].reshape(-1,3)
            colors_cluster = np.repeat(colors_cluster, num_vis, axis=0)[:num_vis]
            labels = labels[:num_vis]
            vis_tracks_3d(server, xyz_np[mask][:num_vis], labels, name=f"cluster_tracks_{i}", colors=colors_cluster)
        """
        
        times =  torch.linspace(-1, 1, T).reshape(1,T,1).repeat(N,1,1)
        xyz_interp = torch.cat((xyz_interp, times), -1)
        inducing_points = xyz_interp[closest_indices, ::time_gap].reshape(-1, 4)
        print(inducing_points.shape)
        
    elif method == 'RX-grid': # random from grid, grid T        
        # Use grid-based approach
        x_min, x_max = args.gp.inducing_min, args.gp.inducing_max
        y_min, y_max = args.gp.inducing_min, args.gp.inducing_max
        z_min, z_max = args.gp.inducing_min, args.gp.inducing_max
        
        # Calculate grid dimensions
        n_spatial = 2

        inducing_points_x = create_grid_inducing_points(
            (x_min, x_max), n_spatial,
            (y_min, y_max), n_spatial,
            (z_min, z_max), n_spatial,
        )


        inducing_points_t = torch.linspace(-1, 1, args.gp.nt)
        expanded_x = inducing_points_x.unsqueeze(1).expand(-1, args.gp.nt, -1)
        expanded_t = inducing_points_t.unsqueeze(0).unsqueeze(2).expand(n_spatial**3, -1, 1)

        inducing_points = torch.cat((expanded_x, expanded_t), dim=-1)

        inducing_points = inducing_points.reshape(-1,4)


    elif method == 'RS-grid': # random sampling from X(xyz), grid T
        pass


    elif method == 'grid':
        # Use grid-based approach
        x_min, x_max = inducing_min, inducing_max
        y_min, y_max = inducing_min, inducing_max
        z_min, z_max = inducing_min, inducing_max
        t_min, t_max = inducing_min, inducing_max
        
        # Calculate grid dimensions
        # TODO
        total_points = num_inducing
        nt = max(4, int(total_points ** 0.25))
        spatial_points = total_points // nt
        n_spatial = int(spatial_points ** (1/3))
        n_spatial = 3
        nt = 100
        
        inducing_points = create_grid_inducing_points(
            (x_min, x_max), (y_min, y_max), (z_min, z_max), (t_min, t_max),
            n_spatial, n_spatial, n_spatial, nt
        )
    else:
        indices = torch.randperm(train_x.size(0))[:num_inducing]
        inducing_points = train_x[indices]
    
    if args.gp.inducing_point_noise_scale > 0: 
        noise = torch.randn_like(inducing_points) * args.gp.inducing_point_noise_scale
        inducing_points = inducing_points + noise

    torch.cuda.empty_cache()
    gc.collect()
    return inducing_points


#def viz_inducing_poitns(inducing_points, bin=30):
#    breakpoint()
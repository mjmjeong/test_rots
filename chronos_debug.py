import torch
import time
from chronos import ChronosPipeline
def get_chronos_embeddings(traj, chunk_size=5000, concat=False):
    
    embeddings_ = {}
    embeddings_['x'], embeddings_['y'], embeddings_['z'] = [], [], []
    total_step = (len(traj) //chunk_size)+1
    mapping = {'x' : 0, 'y' : 1, 'z' : 2}
    traj = torch.tensor(traj, dtype=torch.float32)
    pipeline = ChronosPipeline.from_pretrained(
                "amazon/chronos-t5-small",
                device_map="cuda",
                torch_dtype=torch.bfloat16,
            )

    with torch.no_grad():
        for coord in ['x', 'y', 'z']:
            for step in range(total_step):
                index = mapping[coord]
                try:
                    context = traj[step*chunk_size:(step+1)*chunk_size, :, index]
                except:
                    context = traj[step*chunk_size:, :, index]

                embeddings, _ = pipeline.embed(context)
                embeddings_[coord].append(embeddings)
            embeddings_[coord] = torch.cat(embeddings_[coord], 0)
        if concat:
            embeddings_ = torch.cat([embeddings_['x'], embeddings_['y'], embeddings_['z']], -1)
    return embeddings_
    
qw = torch.rand(3000, 245, 3)
qwe = time.time()
get_chronos_embeddings(qw)
print(time.time()-qwe)
breakpoint()

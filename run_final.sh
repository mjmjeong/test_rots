model_type=IndependentVariationalGPModel

# learning rate
# inducing num 
# inducing random 
# batch structure


# latent G

# visualizaed all instructed
lr=$1
echo "gp lr:" $lr

batch=$2 
echo "batch_size:" $batch

inducing_num=$3
echo "inducing_num:" $inducing_num
#expname=search/debug-b${batch}-lr${lr}-num${inducing_num}

echo "expname" ${4} 
expname=search/debug-${4}
echo $expname

echo "GPU:" ${5}


lr=0.01

lxy=0.001
lxt=0.001
batch=5000
inducing_num=100
expname=search/final_debug-resume-save

GPU=1
#CUDA_VISIBLE_DEVICES=${GPU} python debug_gp.py --exp_name $expname --gp.rots-model $model_type --gp.transls-model $model_type --gp.transls-gp-lr $lr --gp.rots-gp-lr $lr --gp.inducing_num $inducing_num --gp.epochs 3 --gp.x_rsample none  --gp.batch_size $batch --gp.confidence_thred 0.5 
CUDA_VISIBLE_DEVICES=${GPU} python debug_gp.py --exp_name $expname 
#--gp.rots-model $model_type --gp.transls-model $model_type --gp.transls-gp-lr $lr --gp.rots-gp-lr $lr --gp.inducing_num $inducing_num --gp.epochs 3 --gp.x_rsample none  --gp.batch_size $batch --gp.confidence_thred 0.5 
#--work-dir debug_dir --port 8890 
#data:iphone --data.data-dir /131_data/datasets/iphone/spin --data.load_from_cache  

#--gp.rots_gp_lr=0.0010655748484382909 --gp.rots_lengthscale_xy=0.000291302485734549 --gp.rots_lengthscale_zt=0.004338610127395237 --gp.rsample_std=0.05 --gp.transls_gp_lr=0.00014243353649168598 --gp.transls_lengthscale_xy=0.0014437145411310005 --gp.transls_lengthscale_zt=2.0760125450460825e-05 --num_epochs=500 --project=Flow3D_GP_Sweeps

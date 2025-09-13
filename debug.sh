# cfg.lr.w_kl_rots 
# cfg.lr.w_kl_transls 

# TODO: 3DGS vs 2DGS
# TODO: addting bayesian
# TODO: time series initialization (optimization)

var_init=${1}
act=${2}
var_lr=${3}
kl_loss=${4}

exp_name=debug_3dgs_spin_gp
echo "exp_name:" $exp_name
mkdir -p outputs/$exp_name/checkpoints

model_type=IndependentVariationalGPModel

#cp outputs/init_checkpoints/3dgs_init.ckpt outputs/$exp_name/checkpoints/last.ckpt
#--vis-debug \
python run_training.py \
  --work-dir outputs/${exp_name} --exp_name $exp_name --vis-debug \
 --motion.filling_missing_tracks3d gp --gp.rots-model $model_type --gp.transls-model $model_type --gp.inducing_num 50 --gp.input_rsample fix --gp.inner-batch_size 5000 --gp.confidence_thred 0.8 --gp.inner-epochs 100 \
  --motion.rot-type 6d --motion.init-rot-option 6d --motion.basis_type default --tags 3dgs baseline spin --project debug --port 8890 \
  data:iphone \
  --data.data-dir /131_data/datasets/nerf_data/video_nerf/iphone_som/spin \
  --data.load_from_cache

  

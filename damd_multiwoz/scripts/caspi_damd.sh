while [[ "$#" -gt 0 ]]; do
    case $1 in
    	--cuda) cuda="$2"; shift ;;
    	--seed) seed=$2; shift ;;
        --K) K="$2"; shift ;;
        --gamma) gamma=$2; shift ;;
        --policy_loss) policy_loss=$2; shift ;;
        --action_space) action_space=$2; shift ;;
        --metric) metric=$2; shift ;;
        --train_e2e) train_e2e=$2; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

enable_aspn=True

data_file=data_for_damd.json

if [ ${train_e2e} == 'True' ]; then
	bspn_mode=bspn
	enable_dst=True
	use_true_curr_bspn=False
	echo 'training e2e'
else
	bspn_mode=bsdx
	enable_dst=False
	use_true_curr_bspn=True
	echo 'training partial'
fi

if [ ${policy_loss} == 'L_baseline' ]; then
	early_stop_count=5
else
	early_stop_count=6
fi

root_path=./damd_multiwoz

fn_Gs_file_path_dp=${root_path}/data/multi-woz-oppe/fn_Gs_${K}_${gamma}_${action_space}_${metric}.json
fn_Qs_file_path_dp=${root_path}/data/multi-woz-oppe/fn_Qs_${K}_${gamma}_${action_space}_${metric}.json

exp_name=caspi_damd_train_e2e_K_${K}_gamma_${gamma}_${train_e2e}_policy_loss_${policy_loss}_action_space_${action_space}_seed_${seed}

log_file=${exp_name}.log
log_path=${root_path}/logs/${log_file}


python ${root_path}/model.py -mode train -cfg seed=$seed cuda_device=${cuda} \
	exp_no=no_aug batch_size=128 multi_acts_training=False \
	bspn_mode=${bspn_mode} \
	enable_dst=${enable_dst} \
	use_true_curr_bspn=${use_true_curr_bspn} \
	exp_name=${exp_name} \
	fn_Gs_file_path_dp=${fn_Gs_file_path_dp} \
	fn_Qs_file_path_dp=${fn_Qs_file_path_dp} \
	policy_loss=${policy_loss} \
	enable_aspn=${enable_aspn} \
	early_stop_count=${early_stop_count} \
	data_file=${data_file} 


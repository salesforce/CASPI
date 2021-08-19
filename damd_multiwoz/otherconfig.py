other_config = {}

other_config['exp_name']='IS_baseline'
other_config['gen_per_epoch_report']=False
other_config['per_epoch_report_path']=''
other_config['soft_acc']=False
other_config['action_space']={
    0:'act',
    1:'resp'}[0]
other_config['spi_loss_wt']=1.
other_config['spi_const_wt']=0.1
other_config['spi_penalty_coeff']=.2
other_config['policy_loss']={
    0:'L_baseline',
    1:'L_det',
    2:'L_det,L_sto'}[0]
other_config['fn_Gs_file_path_dp']=None#'data/multi-woz-oppe/fn_Gs_dp.json'
other_config['fn_Qs_file_path_dp']=None#'data/multi-woz-oppe/fn_Qs_dp.json'

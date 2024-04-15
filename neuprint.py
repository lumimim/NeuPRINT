import numpy as np
import torch
from dynamical_models import time_invariant_permutation_invariant_linear_recon
from dynamical_models import time_invariant_permutation_invariant_nonlinear_recon
from dynamical_models import time_invariant_permutation_invariant_rnn_recon
from dynamical_models import time_invariant_permutation_invariant_transformer_recon
from dataset import Activity_All_Sessions_Dataset
from loss_function import logLikelihoodGaussian, logLikelihoodPoisson, mse_loss
from classification import post_hoc_subclass_classification
from label_generation import record_id_subclass_label_generation
from torch.nn.parameter import Parameter

class time_permutation_invariant_representation(torch.nn.Module):
    def __init__(self, max_neuron_dim, embedding_dim):
        super(time_permutation_invariant_representation, self).__init__()
        self.embedding = Parameter(torch.FloatTensor(max_neuron_dim, embedding_dim).uniform_(-1, 1), requires_grad = True)
    
    def forward(self):
        return self.embedding

def eval_single_neuron_time_invariant_permutation_invariant_recon_classfier(
    data_all_sessions,
    use_population = False,
    train_val_dataset_ids = None,
    test_dataset_ids = None,
    logging = None,
    writer = None,
    population_keys = ['running_speed', 'eye_size', 'frame_states', 'mean_activity','std_activity'], 
    use_neighbor = False,
    output_distribution = 'mse', 
    use_z = 'True',
    recon_model_type = 'linear',
    classify_model_type = 'linear',
    feature_type = 'embedding', 
    return_last = True,
    all_sessions_neuron_count = 2481,
    train_time_steps_frac = 0.75,
    num_batch = 10,
    batch_size = 128, 
    window_len = 200,
    record_frac = 0.6,
    num_sample_neurons = 80,
    dt = 0.25,
    hidden_dim = 32, 
    post_hoc_hidden_dim = 32,
    embedding_dim = 64, 
    population_feature_dim = 0,
    neighbor_feature_dim = 0,
    layer_dim = 1, 
    nhead = 2,
    mask = None,
    mask_ratio = 0.25,
    mask_last_step_only = False,
    lr = 1e-3,
    lr_scheduler = False,
    z_optimize_lr = 1e-3,
    post_lr = 1e-4,
    num_epochs = 100,
    num_z_optimize_epochs = 50,
    num_post_hoc_epochs = 500,
    post_hoc_eval_epoch_freq = 10,
    max_norm = 1, 
    knn_k = 5,
    k = 1,
    kfold = 5,
    w_clf = 1,
    w_recon = 1,
    use_weighted_ce_loss = False,
    full_context = True,
    full_causal = False,
    context_backward = -1,
    context_forward = -1,
    attend_to_self = True,
    ckpt_path = None,
    use_ckpt_hparams = True,
    post_hoc_only = False):
  
    hparams_dict = locals()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'; print('Using device: %s'%device)
    
    class_all_sessions, _ = record_id_subclass_label_generation(data_all_sessions, test_dataset_ids)
    print('all_sessions_neuron_count:', all_sessions_neuron_count)
    record_unique_ids_all_sessions = class_all_sessions['record_unique_ids_all_sessions'] ### add

    if use_population:
        population_feature_dim = len(population_keys)
    if use_neighbor:
        neighbor_feature_dim = 2
    if recon_model_type == 'linear':
        recon_model = time_invariant_permutation_invariant_linear_recon(
          neuron_dim = num_sample_neurons,
          time_dim = window_len,
          input_dim = 1 + population_feature_dim + neighbor_feature_dim,
          embedding_dim = embedding_dim,
          feature_type = feature_type,
          use_population = use_population,
          use_neighbor = use_neighbor
      ).to(device)
    elif recon_model_type == 'nonlinear':
        recon_model = time_invariant_permutation_invariant_nonlinear_recon(
          neuron_dim = num_sample_neurons,
          time_dim = window_len,
          input_dim = 1 + population_feature_dim + neighbor_feature_dim,
          embedding_dim = embedding_dim,
          feature_type = feature_type,
          use_population = use_population,
          use_neighbor = use_neighbor
      ).to(device)
    elif recon_model_type == 'rnn':
        recon_model = time_invariant_permutation_invariant_rnn_recon(
          neuron_dim = num_sample_neurons,
          time_dim = window_len,
          input_dim = 1 + population_feature_dim + neighbor_feature_dim,
          hidden_dim = hidden_dim,
          embedding_dim = embedding_dim,
          layer_dim = layer_dim,
          use_population = use_population,
          use_neighbor = use_neighbor
      ).to(device)
    elif recon_model_type == 'transformer':
        recon_model = time_invariant_permutation_invariant_transformer_recon(
          time_dim = window_len,
          input_dim = hidden_dim + population_feature_dim + neighbor_feature_dim,
          hidden_dim = hidden_dim,
          embedding_dim = embedding_dim,
          layer_dim = layer_dim,
          nhead = nhead,
          mask_ratio = mask_ratio,
          mask_last_step_only = mask_last_step_only,
          use_population = use_population,
          full_context = full_context,
          full_causal = full_causal,
          context_backward = context_backward,
          context_forward = context_forward,
          attend_to_self = attend_to_self,
          use_neighbor = use_neighbor
      ).to(device)
        
    print('recon model type:', recon_model_type)

    train_activity_dataset = Activity_All_Sessions_Dataset(
            data = data_all_sessions,
            class_data = class_all_sessions,
            record_unique_ids = record_unique_ids_all_sessions,
            device = device, 
            train_val_dataset_ids = train_val_dataset_ids,
            test_dataset_ids = test_dataset_ids,
            num_batch = num_batch,
            batch_size = batch_size,
            train_time_steps_frac = train_time_steps_frac, 
            record_frac = record_frac,
            window_len = window_len,
            num_sample_neurons = num_sample_neurons, 
            population_keys = population_keys,
            is_Training = True,
            is_posthoc = False,
            data_type = 'train')
    
    valid_activity_dataset = Activity_All_Sessions_Dataset(
            data = data_all_sessions,
            class_data = class_all_sessions,
            record_unique_ids = record_unique_ids_all_sessions,
            device = device,
            train_val_dataset_ids = train_val_dataset_ids,
            test_dataset_ids = test_dataset_ids,
            num_batch = num_batch,
            batch_size = batch_size,
            train_time_steps_frac = train_time_steps_frac,
            record_frac = record_frac,
            window_len = window_len,
            num_sample_neurons = num_sample_neurons, 
            population_keys = population_keys,
            is_Training = False,
            is_posthoc = False,
            data_type = 'train')
    
    max_neuron_dim = all_sessions_neuron_count
    tpinv_representation_F = time_permutation_invariant_representation(max_neuron_dim, embedding_dim).to(device)  # n x e        
    optimizer = torch.optim.Adam(list(recon_model.parameters()) + list(tpinv_representation_F.parameters()), lr = lr)

    # create the dataloader
    train_dl = torch.utils.data.DataLoader(train_activity_dataset, batch_size = 1)
    valid_dl = torch.utils.data.DataLoader(valid_activity_dataset, batch_size = 1)
    
    if lr_scheduler:
        scheduler = StepLR(optimizer, step_size=100, gamma=0.1)

    if ckpt_path is not None:
        logging.info(f'loading checkpoint from {ckpt_path}')
        ckpt_dict = torch.load(ckpt_path)
        recon_model.load_state_dict(ckpt_dict['model_state'])
        optimizer.load_state_dict(ckpt_dict['optimizer_state'])
        if lr_scheduler:
            scheduler.load_state_dict(ckpt_dict['lr_scheduler_state'])
        if use_ckpt_hparams:
            hparams_dict = ckpt_dict['hparams_dict']
            print('hparams_dict replaced by saved checkpoint')
        print(ckpt_dict['extra_states'])

    losses = [] if ckpt_path is None else ckpt_dict['extra_states']['losses']
    val_losses = [] if ckpt_path is None else ckpt_dict['extra_states']['val_losses']
    start_epoch = 0 if ckpt_path is None else ckpt_dict['extra_states']['epoch']
    epochs = [] if ckpt_path is None else ckpt_dict['extra_states']['epochs']
    metrics = {} if ckpt_path is None else ckpt_dict['extra_states']['metrics']
    best_epoch_subclass_train_acc = 0.0 if ckpt_path is None else ckpt_dict['best_epoch_subclass_train_acc']
    best_subclass_valid_acc = 0.0 if ckpt_path is None else ckpt_dict['best_subclass_valid_acc']
    best_epoch_subclass_test_acc = 0.0 if ckpt_path is None else ckpt_dict['best_epoch_subclass_test_acc']
    best_epoch_EI_train_acc = 0.0 if ckpt_path is None else ckpt_dict['best_epoch_EI_train_acc']
    best_EI_valid_acc = 0.0 if ckpt_path is None else ckpt_dict['best_EI_valid_acc']
    best_epoch_EI_test_acc = 0.0 if ckpt_path is None else ckpt_dict['best_epoch_EI_test_acc']

    for epoch in range(start_epoch, num_epochs):
        if not post_hoc_only:
            recon_model.train()
            optimizer.zero_grad()

            # cumulative training loss for this epoch
            train_loss_accu = 0
            valid_loss_accu = 0

            # for each batch...
            for i, train_x_list in enumerate(train_dl, 0):
                train_x_ts = train_x_list[0][0] # b x n x t
                train_pop_x_ts = train_x_list[1][0] # b x p x t
                train_neigh_x_ts = train_x_list[2][0] # b x k x t
                neuron_id_ts = train_x_list[3][0] # n
                if recon_model_type == 'transformer':
                    pred_train_x_recon_mean_ts, pred_train_x_recon_logvar_ts, mask  = recon_model(tpinv_representation_F(), train_x_ts, train_pop_x_ts, train_neigh_x_ts, neuron_id_ts)
                else:
                    pred_train_x_recon_mean_ts, pred_train_x_recon_logvar_ts  = recon_model(tpinv_representation_F(), train_x_ts, train_pop_x_ts, train_neigh_x_ts, neuron_id_ts)
                # Compute Loss
                train_x_ts_loss = train_x_ts
                pred_train_x_recon_mean_ts_loss = pred_train_x_recon_mean_ts
                pred_train_x_recon_logvar_ts_loss = pred_train_x_recon_logvar_ts
                if return_last:
                    train_x_ts_loss = train_x_ts_loss[:,:,-1]
                    if recon_model_type == 'rnn' or recon_model_type == 'transformer':
                        pred_train_x_recon_mean_ts_loss = pred_train_x_recon_mean_ts_loss[:,:,-1]
                        pred_train_x_recon_logvar_ts_loss = pred_train_x_recon_logvar_ts_loss[:,:,-1]
                if output_distribution == 'gaussian':
                    recon_loss = - logLikelihoodGaussian(
                        torch.squeeze(train_x_ts_loss), 
                        torch.squeeze(pred_train_x_recon_mean_ts_loss), 
                        torch.squeeze(pred_train_x_recon_logvar_ts_loss),
                        mask=mask)
                elif output_distribution == 'mse':
                    recon_loss = mse_loss(pred=torch.squeeze(pred_train_x_recon_mean_ts_loss), gt=torch.squeeze(train_x_ts_loss), mask=mask) ###
                else:
                    recon_loss = - logLikelihoodPoisson(
                        torch.squeeze(train_x_ts_loss), 
                        torch.squeeze(pred_train_x_recon_mean_ts_loss) * dt,
                        mask=mask)
                loss = w_recon * recon_loss
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(recon_model.parameters(), max_norm = max_norm)
                optimizer.step()
                train_loss_accu += loss.item()
            
            if lr_scheduler:
                # Update the learning rate
                scheduler.step() ###
            train_loss_accu /= (i+1)
            losses.append(train_loss_accu)
            epochs.append(epoch)
            
            recon_model.eval()
            # for each batch...
            for i, valid_x_list in enumerate(valid_dl, 0):
                valid_x_ts = valid_x_list[0][0] # b x n x t
                valid_pop_x_ts = valid_x_list[1][0] # b x p x t
                valid_neigh_x_ts = valid_x_list[2][0] # b x k x t
                neuron_id_ts = valid_x_list[3][0] # n
                if recon_model_type == 'transformer':
                    pred_valid_x_recon_mean_ts, pred_valid_x_recon_logvar_ts, mask  = recon_model(tpinv_representation_F(), valid_x_ts, valid_pop_x_ts, valid_neigh_x_ts, neuron_id_ts)
                else:
                    pred_valid_x_recon_mean_ts, pred_valid_x_recon_logvar_ts = recon_model(tpinv_representation_F(), valid_x_ts, valid_pop_x_ts, valid_neigh_x_ts, neuron_id_ts)
            
                # compute loss
                valid_x_ts_loss = valid_x_ts
                pred_valid_x_recon_mean_ts_loss = pred_valid_x_recon_mean_ts
                pred_valid_x_recon_logvar_ts_loss = pred_valid_x_recon_logvar_ts
                if return_last:
                    valid_x_ts_loss = valid_x_ts_loss[:,:,-1]
                    if recon_model_type == 'rnn' or recon_model_type == 'transformer': ###
                        pred_valid_x_recon_mean_ts_loss = pred_valid_x_recon_mean_ts_loss[:,:,-1]
                        pred_valid_x_recon_logvar_ts_loss = pred_valid_x_recon_logvar_ts_loss[:,:,-1]
                if output_distribution == 'gaussian':
                    valid_recon_loss = - logLikelihoodGaussian(
                        torch.squeeze(valid_x_ts_loss), 
                        torch.squeeze(pred_valid_x_recon_mean_ts_loss), 
                        torch.squeeze(pred_valid_x_recon_logvar_ts_loss),
                        mask=mask)
                elif output_distribution == 'mse':
                    valid_recon_loss = mse_loss(pred=torch.squeeze(pred_valid_x_recon_mean_ts_loss), gt=torch.squeeze(valid_x_ts_loss), mask=mask) ###
                else:
                    valid_recon_loss = - logLikelihoodPoisson(
                        torch.squeeze(valid_x_ts_loss), 
                        torch.squeeze(pred_valid_x_recon_mean_ts_loss) * dt,
                        mask=mask)
                val_loss = w_recon * valid_recon_loss
                valid_loss_accu += val_loss.item()
            
            valid_loss_accu /= (i+1)
            val_losses.append(valid_loss_accu)

            print(f"optimize z and f for train data, epoch: {epoch}, train loss: {train_loss_accu}, valid loss: {valid_loss_accu}")
            logging.info("optimize z and f for train data, epoch: {}, train loss: {}, valid loss: {}".format(epoch, train_loss_accu, valid_loss_accu))
            writer.add_scalar('Loss/train', train_loss_accu, epoch)
            writer.add_scalar('Loss/valid', valid_loss_accu, epoch)
        
        if epoch == 0 or ((epoch + 1) % post_hoc_eval_epoch_freq == 0) or post_hoc_only:

            ####################### optimize z for train, val, test neurons ############################ 
            
            pred_train_z_ts = optimize_time_permutation_invariant_representation(
                                    data = data_all_sessions,
                                    class_data = class_all_sessions,
                                    train_val_dataset_ids = train_val_dataset_ids,
                                    test_dataset_ids = test_dataset_ids,
                                    logging = logging,
                                    record_unique_ids_all_sessions = record_unique_ids_all_sessions,
                                    recon_model = recon_model,
                                    is_posthoc = True,
                                    data_type = 'train',
                                    recon_model_type = recon_model_type,
                                    max_neuron_dim = max_neuron_dim,
                                    embedding_dim = embedding_dim,
                                    output_distribution = output_distribution,
                                    return_last = return_last,
                                    device = device,
                                    num_epochs = num_z_optimize_epochs,
                                    lr = z_optimize_lr,
                                    w_recon = w_recon,
                                    num_batch = num_batch,
                                    batch_size = batch_size,
                                    train_time_steps_frac = train_time_steps_frac, 
                                    record_frac = record_frac,
                                    window_len = window_len,
                                    num_sample_neurons = num_sample_neurons, 
                                    population_keys = population_keys,
                                    max_norm = max_norm,
                                    dt = dt)
            
            pred_val_z_ts = optimize_time_permutation_invariant_representation(
                                    data = data_all_sessions,
                                    class_data = class_all_sessions,
                                    train_val_dataset_ids = train_val_dataset_ids,
                                    test_dataset_ids = test_dataset_ids,
                                    logging = logging,
                                    record_unique_ids_all_sessions = record_unique_ids_all_sessions,
                                    recon_model = recon_model,
                                    is_posthoc = True,
                                    data_type = 'val',
                                    recon_model_type = recon_model_type,
                                    max_neuron_dim = max_neuron_dim,
                                    embedding_dim = embedding_dim,
                                    output_distribution = output_distribution,
                                    return_last = return_last,
                                    device = device,
                                    num_epochs = num_z_optimize_epochs,
                                    lr = z_optimize_lr,
                                    w_recon = w_recon,
                                    num_batch = num_batch,
                                    batch_size = batch_size,
                                    train_time_steps_frac = train_time_steps_frac, 
                                    record_frac = record_frac,
                                    window_len = window_len,
                                    num_sample_neurons = num_sample_neurons, 
                                    population_keys = population_keys,
                                    max_norm = max_norm,
                                    dt = dt)
            
            pred_test_z_ts = optimize_time_permutation_invariant_representation(
                                    data = data_all_sessions,
                                    class_data = class_all_sessions,
                                    train_val_dataset_ids = train_val_dataset_ids,
                                    test_dataset_ids = test_dataset_ids,
                                    logging = logging,
                                    record_unique_ids_all_sessions = record_unique_ids_all_sessions,
                                    recon_model = recon_model,
                                    is_posthoc = True,
                                    data_type = 'test',
                                    recon_model_type = recon_model_type,
                                    max_neuron_dim = max_neuron_dim,
                                    embedding_dim = embedding_dim,
                                    output_distribution = output_distribution,
                                    return_last = return_last,
                                    device = device,
                                    num_epochs = num_z_optimize_epochs,
                                    lr = z_optimize_lr,
                                    w_recon = w_recon,
                                    num_batch = num_batch,
                                    batch_size = batch_size,
                                    train_time_steps_frac = train_time_steps_frac, 
                                    record_frac = record_frac,
                                    window_len = window_len,
                                    num_sample_neurons = num_sample_neurons, 
                                    population_keys = population_keys,
                                    max_norm = max_norm,
                                    dt = dt)

            ################################# traiing posthoc classifier  ##################################### 
            feature_dim = embedding_dim

            print('-------task: subclass-------')
            print('num of class: ', 5)
            logging.info('-------task: subclass-------')
            logging.info('num of class: {}'.format(5))
            for model_type in ['knn','linear','mlp']:
                sub_train_accs = []
                sub_valid_accs = []
                sub_test_accs = []
                y_train_trues = []
                y_train_preds = []
                y_valid_trues = []
                y_valid_preds = []
                y_test_trues = []
                y_test_preds = []
                train_zs = []
                valid_zs = []
                test_zs = []
                for kf_i in range(kfold):
                    outputs = post_hoc_subclass_classification(
                        class_all_sessions,
                        pred_train_z_ts,
                        pred_val_z_ts,
                        pred_test_z_ts,
                        device = device,
                        num_post_hoc_epochs = num_post_hoc_epochs,
                        task = 'subclass',
                        subclass_labels = ['Lamp5', 'Vip', 'Pvalb', 'Sst', 'Sncg'],
                        classify_model_type = model_type,
                        feature_dim = feature_dim,
                        hidden_dim = post_hoc_hidden_dim,
                        max_norm = max_norm,
                        k = k,
                        post_lr = post_lr,
                        knn_k = knn_k,
                        use_weighted_ce_loss = use_weighted_ce_loss) 
                    sub_train_accs.append(outputs['train_acc'])
                    sub_valid_accs.append(outputs['valid_acc'])
                    sub_test_accs.append(outputs['test_acc'])
                    y_train_trues.append(outputs['y_train_true'])
                    y_train_preds.append(outputs['y_train_pred'])
                    y_valid_trues.append(outputs['y_valid_true'])
                    y_valid_preds.append(outputs['y_valid_pred'])
                    y_test_trues.append(outputs['y_test_true'])
                    y_test_preds.append(outputs['y_test_pred'])
                    train_zs.append(outputs['train_z'])
                    valid_zs.append(outputs['valid_z'])
                    test_zs.append(outputs['test_z'])
        
                print('classify_model_type:', model_type)
                logging.info('classify_model_type:{}'.format(model_type))
                print('valid acc mean:', np.mean(np.asarray(sub_valid_accs)))
                print('valid acc std:', np.std(np.asarray(sub_valid_accs)))
                print('valid acc max:', np.max(np.asarray(sub_valid_accs)))

                print('test acc mean:', np.mean(np.asarray(sub_test_accs)))
                print('test acc std:', np.std(np.asarray(sub_test_accs)))
                print('test acc max:', np.max(np.asarray(sub_test_accs)))

                logging.info('valid acc mean:{}'.format(np.mean(np.asarray(sub_valid_accs))))
                logging.info('valid acc std:{}'.format(np.std(np.asarray(sub_valid_accs))))
                logging.info('valid acc max:{}'.format(np.max(np.asarray(sub_valid_accs))))

                logging.info('test acc mean:{}'.format(np.mean(np.asarray(sub_test_accs))))
                logging.info('test acc std:{}'.format(np.std(np.asarray(sub_test_accs))))
                logging.info('test acc max:{}'.format(np.max(np.asarray(sub_test_accs))))

                writer.add_scalar(f'Accuracy/subclass_{model_type}/valid_mean', np.mean(np.asarray(sub_valid_accs)),epoch)
                writer.add_scalar(f'Accuracy/subclass_{model_type}/valid_max', np.max(np.asarray(sub_valid_accs)),epoch)
                writer.add_scalar(f'Accuracy/subclass_{model_type}/test_mean', np.mean(np.asarray(sub_test_accs)),epoch)
                writer.add_scalar(f'Accuracy/subclass_{model_type}/test_max', np.max(np.asarray(sub_test_accs)),epoch)

                if model_type == classify_model_type and (np.mean(np.asarray(sub_valid_accs)) > best_subclass_valid_acc or post_hoc_only):
                    best_epoch_subclass_train_acc = np.mean(np.asarray(sub_train_accs))
                    best_subclass_valid_acc = np.mean(np.asarray(sub_valid_accs))
                    best_epoch_subclass_test_acc = np.mean(np.asarray(sub_test_accs))
                    metrics['subclass_train_' + model_type] = np.mean(np.asarray(sub_train_accs))
                    metrics['subclass_valid_' + model_type] = np.mean(np.asarray(sub_valid_accs))
                    metrics['subclass_test_' + model_type] = np.mean(np.asarray(sub_test_accs))

                    y_train_trues = np.concatenate(np.asarray(y_train_trues), axis=0)
                    y_train_preds = np.concatenate(np.asarray(y_train_preds), axis=0)
                    train_zs = np.concatenate(np.asarray(train_zs), axis=0)

                    y_valid_trues = np.concatenate(np.asarray(y_valid_trues), axis=0)
                    y_valid_preds = np.concatenate(np.asarray(y_valid_preds), axis=0)
                    valid_zs = np.concatenate(np.asarray(valid_zs), axis=0)

                    y_test_trues = np.concatenate(np.asarray(y_test_trues), axis=0)
                    y_test_preds = np.concatenate(np.asarray(y_test_preds), axis=0)
                    test_zs = np.concatenate(np.asarray(test_zs), axis=0)

                    ckpt_dict = {
                        'model_state': recon_model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'lr_scheduler_state': None if not lr_scheduler else scheduler.state_dict(),
                        'hparams_dict': hparams_dict,
                        'best_epoch_subclass_train_acc': best_epoch_subclass_train_acc,
                        'best_subclass_valid_acc': best_subclass_valid_acc,
                        'best_epoch_subclass_test_acc': best_epoch_subclass_test_acc,
                        'best_epoch_EI_train_acc': best_epoch_EI_train_acc,
                        'best_EI_valid_acc': best_EI_valid_acc,
                        'best_epoch_EI_valid_acc': best_epoch_EI_test_acc,
                        'best_train_embedding': pred_train_z_ts,
                        'best_valid_embedding': pred_val_z_ts,
                        'best_test_embedding': pred_test_z_ts,
                        'y_train_trues': y_train_trues,
                        'y_train_preds': y_train_preds,
                        'train_zs': train_zs,
                        'y_valid_trues': y_valid_trues,
                        'y_valid_preds': y_valid_preds,
                        'valid_zs': valid_zs,
                        'y_test_trues': y_test_trues,
                        'y_test_preds': y_test_preds,
                        'test_zs': test_zs,
                        'extra_states': {
                            'epoch': epoch,
                            'epochs': epochs,
                            'losses': losses,
                            'val_losses': val_losses,
                            'metrics': metrics,
                        }
                    }
                    torch.save(ckpt_dict, os.path.join(ckpt_dir, 'best_subclass.pt'))
            
            print('-------task: EI -------')
            print('num of class: ', 2)
            logging.info('-------task: EI-------')
            logging.info('num of class: {}'.format(2))
            for model_type in ['knn','linear','mlp']:
                EI_train_accs = []
                EI_valid_accs = []
                EI_test_accs = []
                y_train_trues = []
                y_train_preds = []
                y_valid_trues = []
                y_valid_preds = []
                y_test_trues = []
                y_test_preds = []
                train_zs = []
                valid_zs = []
                test_zs = []
                for kf_i in range(kfold):
                    outputs = post_hoc_subclass_classification(
                        class_all_sessions,
                        pred_train_z_ts,
                        pred_val_z_ts,
                        pred_test_z_ts,
                        device = device,
                        num_post_hoc_epochs = num_post_hoc_epochs,
                        task = 'EI',
                        subclass_labels = ['EC', 'IN'],
                        classify_model_type = model_type,
                        feature_dim = feature_dim,
                        hidden_dim = post_hoc_hidden_dim,
                        max_norm = max_norm,
                        k = k,
                        post_lr = post_lr,
                        knn_k = knn_k,
                        use_weighted_ce_loss = use_weighted_ce_loss) 
                    
                    EI_train_accs.append(outputs['train_acc'])
                    EI_valid_accs.append(outputs['valid_acc'])
                    EI_test_accs.append(outputs['test_acc'])
                    y_train_trues.append(outputs['y_train_true'])
                    y_train_preds.append(outputs['y_train_pred'])
                    y_valid_trues.append(outputs['y_valid_true'])
                    y_valid_preds.append(outputs['y_valid_pred'])
                    y_test_trues.append(outputs['y_test_true'])
                    y_test_preds.append(outputs['y_test_pred'])
                    train_zs.append(outputs['train_z'])
                    valid_zs.append(outputs['valid_z'])
                    test_zs.append(outputs['test_z'])

                print('classify_model_type:', model_type)
                logging.info('classify_model_type:{}'.format(model_type))
                print('valid acc mean:', np.mean(np.asarray(EI_valid_accs)))
                print('valid acc std:', np.std(np.asarray(EI_valid_accs)))
                print('valid acc max:', np.max(np.asarray(EI_valid_accs)))

                print('test acc mean:', np.mean(np.asarray(EI_test_accs)))
                print('test acc std:', np.std(np.asarray(EI_test_accs)))
                print('test acc max:', np.max(np.asarray(EI_test_accs)))

                logging.info('valid acc mean:{}'.format(np.mean(np.asarray(EI_valid_accs))))
                logging.info('valid acc std:{}'.format(np.std(np.asarray(EI_valid_accs))))
                logging.info('valid acc max:{}'.format(np.max(np.asarray(EI_valid_accs))))

                logging.info('test acc mean:{}'.format(np.mean(np.asarray(EI_test_accs))))
                logging.info('test acc std:{}'.format(np.std(np.asarray(EI_test_accs))))
                logging.info('test acc max:{}'.format(np.max(np.asarray(EI_test_accs))))

                writer.add_scalar(f'Accuracy/EI_{model_type}/valid_mean', np.mean(np.asarray(EI_valid_accs)), epoch)
                writer.add_scalar(f'Accuracy/EI_{model_type}/valid_max', np.max(np.asarray(EI_valid_accs)), epoch)
                writer.add_scalar(f'Accuracy/EI_{model_type}/test_mean', np.mean(np.asarray(EI_test_accs)), epoch)
                writer.add_scalar(f'Accuracy/EI_{model_type}/test_max', np.max(np.asarray(EI_test_accs)), epoch)

                if model_type == classify_model_type and (np.mean(np.asarray(EI_valid_accs)) > best_EI_valid_acc or post_hoc_only):
                    best_epoch_EI_train_acc = np.mean(np.asarray(EI_train_accs))
                    best_EI_valid_acc = np.mean(np.asarray(EI_valid_accs))
                    best_epoch_EI_test_acc = np.mean(np.asarray(EI_test_accs))

                    metrics['EI_train_' + model_type] = np.mean(np.asarray(EI_train_accs))
                    metrics['EI_valid_' + model_type] = np.mean(np.asarray(EI_valid_accs))
                    metrics['EI_test_' + model_type] = np.mean(np.asarray(EI_test_accs))

                    y_train_trues = np.concatenate(np.asarray(y_train_trues), axis=0)
                    y_train_preds = np.concatenate(np.asarray(y_train_preds), axis=0)
                    train_zs = np.concatenate(np.asarray(train_zs), axis=0)

                    y_valid_trues = np.concatenate(np.asarray(y_valid_trues), axis=0)
                    y_valid_preds = np.concatenate(np.asarray(y_valid_preds), axis=0)
                    valid_zs = np.concatenate(np.asarray(valid_zs), axis=0)

                    y_test_trues = np.concatenate(np.asarray(y_test_trues), axis=0)
                    y_test_preds = np.concatenate(np.asarray(y_test_preds), axis=0)
                    test_zs = np.concatenate(np.asarray(test_zs), axis=0)

                    ckpt_dict = {
                        'model_state': recon_model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'lr_scheduler_state': None if not lr_scheduler else scheduler.state_dict(),
                        'hparams_dict': hparams_dict,
                        'best_epoch_subclass_train_acc': best_epoch_subclass_train_acc,
                        'best_subclass_valid_acc': best_subclass_valid_acc,
                        'best_epoch_subclass_test_acc': best_epoch_subclass_test_acc,
                        'best_epoch_EI_train_acc': best_epoch_EI_train_acc,
                        'best_EI_valid_acc': best_EI_valid_acc,
                        'best_epoch_EI_valid_acc': best_epoch_EI_test_acc,
                        'best_train_embedding': pred_train_z_ts,
                        'best_valid_embedding': pred_val_z_ts,
                        'best_test_embedding': pred_test_z_ts,
                        'y_train_trues': y_train_trues,
                        'y_train_preds': y_train_preds,
                        'train_zs': train_zs,
                        'y_valid_trues': y_valid_trues,
                        'y_valid_preds': y_valid_preds,
                        'valid_zs': valid_zs,
                        'y_test_trues': y_test_trues,
                        'y_test_preds': y_test_preds,
                        'test_zs': test_zs,
                        'extra_states': {
                            'epoch': epoch,
                            'epochs': epochs,
                            'losses': losses,
                            'val_losses': val_losses,
                            'metrics': metrics,
                        }
                    }

                    torch.save(ckpt_dict, os.path.join(ckpt_dir, 'best_EI.pt'))

            if post_hoc_only:

                outputs = {
                    'best_train_embedding': pred_train_z_ts,
                    'best_valid_embedding': pred_val_z_ts,
                    'best_test_embedding': pred_test_z_ts,
                    'metrics': metrics,
                }
                return outputs

    if use_z:
        outputs = {
            'epochs': epochs,
            'losses': losses,
            'val_loss': val_loss,
            'train_x': train_x_ts.detach().cpu(),
            'valid_x': valid_x_ts.detach().cpu(),
            'pred_train_z': pred_train_z_ts.detach().cpu(),
            'recon_train_x_mean': pred_train_x_recon_mean_ts.detach().cpu(),
            'recon_train_x_logvar': pred_train_x_recon_logvar_ts.detach().cpu(),
            'recon_valid_x_mean': pred_valid_x_recon_mean_ts.detach().cpu(),
            'recon_valid_x_logvar': pred_train_x_recon_logvar_ts.detach().cpu(),
            'metrics': metrics,
        }
    else:
        outputs = {
            'train_x': train_x_ts.detach().cpu(),
            'valid_x': valid_x_ts.detach().cpu(),
        }

    return outputs

def optimize_time_permutation_invariant_representation(
        data,
        class_data,
        train_val_dataset_ids,
        test_dataset_ids,
        logging,
        record_unique_ids_all_sessions,
        recon_model,
        is_posthoc,
        data_type,
        recon_model_type,
        max_neuron_dim,
        embedding_dim,
        output_distribution,
        return_last,
        device,
        num_epochs,
        lr,
        w_recon,
        num_batch,
        batch_size,
        train_time_steps_frac, 
        record_frac,
        window_len,
        num_sample_neurons, 
        population_keys,
        max_norm,
        dt,
        mask = None):
    
    print('data_type:', data_type)
    
    tpinv_representation =  time_permutation_invariant_representation(max_neuron_dim, embedding_dim).to(device) # n x e
    optimizer = torch.optim.Adam(tpinv_representation.parameters(), lr = lr)

    train_activity_dataset = Activity_All_Sessions_Dataset(
            data = data,
            class_data = class_data,
            record_unique_ids = record_unique_ids_all_sessions,
            device = device, 
            train_val_dataset_ids = train_val_dataset_ids,
            test_dataset_ids = test_dataset_ids,
            num_batch = num_batch,
            batch_size = batch_size,
            train_time_steps_frac = train_time_steps_frac, 
            record_frac = record_frac,
            window_len = window_len,
            num_sample_neurons = num_sample_neurons, 
            population_keys = population_keys,
            is_Training = True,
            is_posthoc = is_posthoc,
            data_type = data_type)
    
    valid_activity_dataset = Activity_All_Sessions_Dataset(
            data = data,
            class_data = class_data,
            record_unique_ids = record_unique_ids_all_sessions,
            train_val_dataset_ids = train_val_dataset_ids,
            test_dataset_ids = test_dataset_ids,
            device = device, 
            num_batch = num_batch,
            batch_size = batch_size,
            train_time_steps_frac = train_time_steps_frac,
            record_frac = record_frac,
            window_len = window_len,
            num_sample_neurons = num_sample_neurons, 
            population_keys = population_keys,
            is_Training = False,
            is_posthoc = is_posthoc,
            data_type = data_type)
    
    # create the dataloader
    train_dl = torch.utils.data.DataLoader(train_activity_dataset, batch_size = 1)
    valid_dl = torch.utils.data.DataLoader(valid_activity_dataset, batch_size = 1)

    losses = []
    val_losses = []
    epochs = []

    for epoch in range(num_epochs):
        recon_model.train()
        optimizer.zero_grad()

        # cumulative training loss for this epoch
        train_loss_accu = 0
        valid_loss_accu = 0

        # for each batch...
        for i, train_x_list in enumerate(train_dl, 0):
            train_x_ts = train_x_list[0][0] # b x n x t
            train_pop_x_ts = train_x_list[1][0] # b x p x t
            train_neigh_x_ts = train_x_list[2][0] # b x k x t
            neuron_id_ts = train_x_list[3][0] # n
            if recon_model_type == 'transformer':
                pred_train_x_recon_mean_ts, pred_train_x_recon_logvar_ts, mask  = recon_model(tpinv_representation(), train_x_ts, train_pop_x_ts, train_neigh_x_ts, neuron_id_ts)
            else:
                pred_train_x_recon_mean_ts, pred_train_x_recon_logvar_ts  = recon_model(tpinv_representation(), train_x_ts, train_pop_x_ts, train_neigh_x_ts, neuron_id_ts) 
            # Compute Loss
            train_x_ts_loss = train_x_ts
            pred_train_x_recon_mean_ts_loss = pred_train_x_recon_mean_ts
            pred_train_x_recon_logvar_ts_loss = pred_train_x_recon_logvar_ts
            if return_last:
                train_x_ts_loss = train_x_ts_loss[:,:,-1]
                if recon_model_type == 'rnn' or recon_model_type == 'transformer':
                    pred_train_x_recon_mean_ts_loss = pred_train_x_recon_mean_ts_loss[:,:,-1]
                    pred_train_x_recon_logvar_ts_loss = pred_train_x_recon_logvar_ts_loss[:,:,-1]
            if output_distribution == 'gaussian':
                recon_loss = - logLikelihoodGaussian(
                    torch.squeeze(train_x_ts_loss), 
                    torch.squeeze(pred_train_x_recon_mean_ts_loss), 
                    torch.squeeze(pred_train_x_recon_logvar_ts_loss),
                    mask=mask)
            elif output_distribution == 'mse':
                recon_loss = mse_loss(pred=torch.squeeze(pred_train_x_recon_mean_ts_loss), gt=torch.squeeze(train_x_ts_loss), mask=mask) ###
            else:
                recon_loss = - logLikelihoodPoisson(
                    torch.squeeze(train_x_ts_loss), 
                    torch.squeeze(pred_train_x_recon_mean_ts_loss) * dt,
                    mask=mask)
            loss = w_recon * recon_loss
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(recon_model.parameters(), max_norm = max_norm)
            optimizer.step()
            train_loss_accu += loss.item()
        
        train_loss_accu /= (i+1)
        losses.append(train_loss_accu)
        epochs.append(epoch)
        
        recon_model.eval()
        # for each batch...
        for i, valid_x_list in enumerate(valid_dl, 0):
            valid_x_ts = valid_x_list[0][0] # b x n x t
            valid_pop_x_ts = valid_x_list[1][0] # b x p x t
            valid_neigh_x_ts = valid_x_list[2][0] # b x k x t
            neuron_id_ts = valid_x_list[3][0] # n
            if recon_model_type == 'transformer':
                pred_valid_x_recon_mean_ts, pred_valid_x_recon_logvar_ts, mask  = recon_model(tpinv_representation(), valid_x_ts, valid_pop_x_ts, valid_neigh_x_ts, neuron_id_ts)
            else:
                pred_valid_x_recon_mean_ts, pred_valid_x_recon_logvar_ts = recon_model(tpinv_representation(), valid_x_ts, valid_pop_x_ts, valid_neigh_x_ts, neuron_id_ts)
        
            # compute loss
            valid_x_ts_loss = valid_x_ts
            pred_valid_x_recon_mean_ts_loss = pred_valid_x_recon_mean_ts
            pred_valid_x_recon_logvar_ts_loss = pred_valid_x_recon_logvar_ts
            if return_last:
                valid_x_ts_loss = valid_x_ts_loss[:,:,-1]
                if recon_model_type == 'rnn' or recon_model_type == 'transformer': ###
                    pred_valid_x_recon_mean_ts_loss = pred_valid_x_recon_mean_ts_loss[:,:,-1]
                    pred_valid_x_recon_logvar_ts_loss = pred_valid_x_recon_logvar_ts_loss[:,:,-1]
            if output_distribution == 'gaussian':
                valid_recon_loss = - logLikelihoodGaussian(
                    torch.squeeze(valid_x_ts_loss), 
                    torch.squeeze(pred_valid_x_recon_mean_ts_loss), 
                    torch.squeeze(pred_valid_x_recon_logvar_ts_loss),
                    mask=mask)
            elif output_distribution == 'mse':
                valid_recon_loss = mse_loss(pred=torch.squeeze(pred_valid_x_recon_mean_ts_loss), gt=torch.squeeze(valid_x_ts_loss), mask=mask) ###
            else:
                valid_recon_loss = - logLikelihoodPoisson(
                    torch.squeeze(valid_x_ts_loss), 
                    torch.squeeze(pred_valid_x_recon_mean_ts_loss) * dt,
                    mask=mask)
            val_loss = w_recon * valid_recon_loss
            valid_loss_accu += val_loss.item()
        
        valid_loss_accu /= (i+1)
        val_losses.append(valid_loss_accu)

        print(f"optimize z for {data_type} neuron: epoch: {epoch}, train loss: {train_loss_accu}, valid loss: {valid_loss_accu}")
        logging.info("optimize z for {} neuron: epoch: {}, train loss: {}, valid loss: {}".format(data_type, epoch, train_loss_accu, valid_loss_accu))

    return tpinv_representation().detach()
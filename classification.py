import torch
import numpy as np
from classifiers import linear_classifier, mlp_classifier, mlp_classifier6

def post_hoc_subclass_classification(
    class_all_sessions,
    pred_train_z_ts,
    pred_val_z_ts,
    pred_test_z_ts,
    device,
    num_post_hoc_epochs = 500,
    task = 'subclass',
    subclass_labels = ['Lamp5', 'Sncg', 'Pvalb', 'Vip', 'Sst'],
    classify_model_type = 'mlp',
    feature_dim = 8,
    hidden_dim = 8,
    max_norm = 1,
    k = 1,
    post_lr = 1e-4,
    knn_k = 5,
    use_weighted_ce_loss = False,
    figure_vis = False):
    # post-hoc linear classification, neuron holdout evaluation
    
    num_classes = len(subclass_labels)

    if classify_model_type == 'linear':
        class_classifier = linear_classifier(
            input_dim = feature_dim,
            output_dim = num_classes).to(device)
    elif classify_model_type == 'mlp':
        class_classifier = mlp_classifier(
            input_dim = feature_dim,
            hidden_dim = hidden_dim,
            output_dim = num_classes).to(device)
    else:
        class_classifier = mlp_classifier6(
            input_dim = feature_dim,
            hidden_dim = hidden_dim,
            output_dim = num_classes).to(device)

    if task == 'subclass':
        record_subclass_all_sessions = class_all_sessions['record_subclass_all_sessions']
        record_unique_ids_all_sessions = class_all_sessions['record_unique_ids_all_sessions']
        record_train_neuron_unique_ids = class_all_sessions['record_train_neuron_unique_ids']
        record_val_neuron_unique_ids = class_all_sessions['record_val_neuron_unique_ids']
        record_test_neuron_unique_ids = class_all_sessions['record_test_neuron_unique_ids']
    else:
        record_subclass_all_sessions = class_all_sessions['neuron_EI_class_all_sessions']
        record_unique_ids_all_sessions = class_all_sessions['neuron_unique_ids_all_sessions']
        record_train_neuron_unique_ids = class_all_sessions['EI_train_neuron_unique_ids']
        record_val_neuron_unique_ids = class_all_sessions['EI_val_neuron_unique_ids']
        record_test_neuron_unique_ids = class_all_sessions['EI_test_neuron_unique_ids']

    train_z_list = []
    valid_z_list = []
    test_z_list = []
    train_y_list = []
    valid_y_list = []
    test_y_list = []
    train_class_counts = []
    valid_class_counts = []
    test_class_counts = []
    for i, subclass_label in enumerate(subclass_labels):
        class_indexes = record_unique_ids_all_sessions[np.where(record_subclass_all_sessions == subclass_label)]
        train_class_indexes = np.asarray(list(set(class_indexes.tolist()) & set(record_train_neuron_unique_ids.tolist()))).astype(int)
        valid_class_indexes = np.asarray(list(set(class_indexes.tolist()) & set(record_val_neuron_unique_ids.tolist()))).astype(int)
        test_class_indexes = np.asarray(list(set(class_indexes.tolist()) & set(record_test_neuron_unique_ids.tolist()))).astype(int)
        
        train_z_list.append(pred_train_z_ts[train_class_indexes, :].detach().cpu())
        valid_z_list.append(pred_val_z_ts[valid_class_indexes, :].detach().cpu())
        test_z_list.append(pred_test_z_ts[test_class_indexes, :].detach().cpu())

        train_y_label = np.zeros((train_class_indexes.shape[0], num_classes))
        valid_y_label = np.zeros((valid_class_indexes.shape[0], num_classes))
        test_y_label = np.zeros((test_class_indexes.shape[0], num_classes))

        train_y_label[:, i] = 1
        valid_y_label[:, i] = 1
        test_y_label[:, i] = 1

        train_y_list.append(train_y_label)
        valid_y_list.append(valid_y_label)
        test_y_list.append(test_y_label)

        train_class_counts.append(train_class_indexes.shape[0])
        valid_class_counts.append(valid_class_indexes.shape[0])
        test_class_counts.append(test_class_indexes.shape[0])

    train_z = np.expand_dims(np.concatenate(train_z_list, axis = 0), axis = 0)
    valid_z = np.expand_dims(np.concatenate(valid_z_list, axis = 0), axis = 0)
    test_z = np.expand_dims(np.concatenate(test_z_list, axis = 0), axis = 0)

    train_y = np.expand_dims(np.concatenate(train_y_list, axis = 0), axis = 0)
    valid_y = np.expand_dims(np.concatenate(valid_y_list, axis = 0), axis = 0)
    test_y = np.expand_dims(np.concatenate(test_y_list, axis = 0), axis = 0)

    train_z_ts = torch.Tensor(train_z).to(device)
    valid_z_ts = torch.Tensor(valid_z).to(device)
    test_z_ts = torch.Tensor(test_z).to(device)

    train_y_ts = torch.Tensor(train_y).to(device)
    valid_y_ts = torch.Tensor(valid_y).to(device)
    test_y_ts = torch.Tensor(test_y).to(device)
    
    if classify_model_type == 'knn':
        # post-hoc knn classification, neuron holdout evaluation
        num_trial = train_z_ts.shape[0]

        num_neuron = train_z_ts.shape[1]
        num_feature_knn = train_z_ts.detach().cpu().shape[2]
        train_knn_z = np.reshape(train_z_ts.detach().cpu().numpy(), (num_trial * num_neuron, num_feature_knn))
        train_knn_y = np.reshape(np.repeat(np.argmax(train_y_ts.detach().cpu()[0, :, :], axis = 1)[np.newaxis, :], num_trial, axis = 0).numpy(), (num_trial * num_neuron, 1))

        valid_num_neuron = valid_z_ts.shape[1]
        valid_knn_z = np.reshape(valid_z_ts.detach().cpu().numpy(), (num_trial * valid_num_neuron, num_feature_knn))
        valid_knn_y = np.reshape(np.repeat(np.argmax(valid_y_ts.detach().cpu()[0, :, :], axis = 1)[np.newaxis, :], num_trial, axis = 0).numpy(), (num_trial * valid_num_neuron, 1))

        test_num_neuron = test_z_ts.shape[1]
        test_knn_z = np.reshape(test_z_ts.detach().cpu().numpy(), (num_trial * test_num_neuron, num_feature_knn))
        test_knn_y = np.reshape(np.repeat(np.argmax(test_y_ts.detach().cpu()[0, :, :], axis = 1)[np.newaxis, :], num_trial, axis = 0).numpy(), (num_trial * test_num_neuron, 1))

        neigh = KNeighborsClassifier(n_neighbors = knn_k)
        neigh.fit(train_knn_z, np.squeeze(train_knn_y))
        train_knn_pred = neigh.predict(train_knn_z)
        valid_knn_pred = neigh.predict(valid_knn_z)
        test_knn_pred = neigh.predict(test_knn_z)

        post_hoc_knn_train_accs = neigh.score(train_knn_z, train_knn_y)
        post_hoc_knn_valid_acc = neigh.score(valid_knn_z, valid_knn_y)
        post_hoc_knn_test_acc = neigh.score(test_knn_z, test_knn_y)

        outputs = {
            'train_acc': post_hoc_knn_train_accs,
            'valid_acc': post_hoc_knn_valid_acc,
            'test_acc': post_hoc_knn_test_acc,
            'y_train_true': train_knn_y,
            'y_train_pred': train_knn_pred,
            'y_valid_true': valid_knn_y,
            'y_valid_pred': valid_knn_pred,
            'y_test_true': test_knn_y,
            'y_test_pred': test_knn_pred,
            'train_z': train_knn_z,
            'valid_z': valid_knn_z,
            'test_z': test_knn_z
        }

        return outputs

    else:
        if task == 'subclass':
            weights_raw = [1,1,1,3,3]
        else:
            weights_raw = np.sum(train_class_counts)/train_class_counts
        weights_raw /= np.sum(weights_raw)
        train_class_weights = torch.Tensor(weights_raw).to(device)
        ce_weight = train_class_weights if use_weighted_ce_loss else None
        clf_criterion = torch.nn.CrossEntropyLoss(reduction = 'none', weight=ce_weight)
        post_hoc_optimizer = torch.optim.Adam(class_classifier.parameters(), lr = post_lr)
        post_hoc_scheduler = StepLR(post_hoc_optimizer, step_size=2000, gamma=0.5)
        post_hoc_clf_losses = []
        post_hoc_accs = []
        post_hoc_valid_clf_losses = []
        post_hoc_valid_accs = []
        post_hoc_test_clf_losses = []
        post_hoc_test_accs = []
        post_hoc_epochs = []
        
        best_val_acc = 0.0

        for epoch in range(num_post_hoc_epochs):
            class_classifier.train()
            post_hoc_optimizer.zero_grad()
            # Forward pass
            pred_post_hoc_train_y_ts = class_classifier(train_z_ts)
            # Compute Loss
            pred_post_hoc_train_y_ts_reshape = torch.reshape(pred_post_hoc_train_y_ts, (pred_post_hoc_train_y_ts.shape[0] * pred_post_hoc_train_y_ts.shape[1], pred_post_hoc_train_y_ts.shape[2]))
            train_y_ts_reshape = torch.reshape(train_y_ts, (train_y_ts.shape[0] * train_y_ts.shape[1], train_y_ts.shape[2]))
            post_hoc_clf_loss_element_reshape = clf_criterion(pred_post_hoc_train_y_ts_reshape, train_y_ts_reshape)
            post_hoc_clf_loss_element = torch.reshape(post_hoc_clf_loss_element_reshape, (train_y_ts.shape[0], train_y_ts.shape[1]))
            post_hoc_clf_loss = torch.mean(post_hoc_clf_loss_element) ### no mask applied
            post_hoc_loss = post_hoc_clf_loss
            _, pred_post_pred_indexes = torch.topk(pred_post_hoc_train_y_ts, k = k, dim = -1)
            repeat_gt = torch.argmax(train_y_ts, dim = -1).unsqueeze(-1).repeat(1,1,k)
            post_hoc_value, _ = torch.max((pred_post_pred_indexes == repeat_gt).int(), dim = -1)
            post_hoc_acc = torch.mean(post_hoc_value.float()) ### no mask applied
            post_hoc_clf_losses.append(post_hoc_clf_loss.item())
            post_hoc_accs.append(post_hoc_acc.item())
            post_hoc_epochs.append(epoch)

            # Backward pass
            post_hoc_loss.backward()
            torch.nn.utils.clip_grad_norm_(class_classifier.parameters(), max_norm = max_norm)
            post_hoc_optimizer.step()
            
            # Update the learning rate
            post_hoc_scheduler.step() ###

            # validation step
            class_classifier.eval()
            # Forward pass
            pred_post_hoc_valid_y_ts = class_classifier(valid_z_ts.detach())
            # Compute Loss
            pred_post_hoc_valid_y_ts_reshape = torch.reshape(pred_post_hoc_valid_y_ts, (pred_post_hoc_valid_y_ts.shape[0] * pred_post_hoc_valid_y_ts.shape[1], pred_post_hoc_valid_y_ts.shape[2]))
            valid_y_ts_reshape = torch.reshape(valid_y_ts, (valid_y_ts.shape[0] * valid_y_ts.shape[1], valid_y_ts.shape[2]))
            post_hoc_valid_clf_loss_element_reshape = clf_criterion(pred_post_hoc_valid_y_ts_reshape, valid_y_ts_reshape)
            post_hoc_valid_clf_loss_element = torch.reshape(post_hoc_valid_clf_loss_element_reshape, (valid_y_ts.shape[0], valid_y_ts.shape[1]))
            post_hoc_valid_clf_loss = torch.mean(post_hoc_valid_clf_loss_element)
            _, pred_post_valid_pred_indexes = torch.topk(pred_post_hoc_valid_y_ts, k = k, dim = -1)
            repeat_gt = torch.argmax(valid_y_ts, dim = -1).unsqueeze(-1).repeat(1,1,k)
            post_hoc_valid_value, _ = torch.max((pred_post_valid_pred_indexes == repeat_gt).int(), dim = -1)
            post_hoc_valid_acc = torch.mean(post_hoc_valid_value.float())
            post_hoc_valid_clf_losses.append(post_hoc_valid_clf_loss.item())
            post_hoc_valid_accs.append(post_hoc_valid_acc.item())

            # test step
            pred_post_hoc_test_y_ts = class_classifier(test_z_ts.detach())
            # Compute Loss
            pred_post_hoc_test_y_ts_reshape = torch.reshape(pred_post_hoc_test_y_ts, (pred_post_hoc_test_y_ts.shape[0] * pred_post_hoc_test_y_ts.shape[1], pred_post_hoc_test_y_ts.shape[2]))
            test_y_ts_reshape = torch.reshape(test_y_ts, (test_y_ts.shape[0] * test_y_ts.shape[1], test_y_ts.shape[2]))
            post_hoc_test_clf_loss_element_reshape = clf_criterion(pred_post_hoc_test_y_ts_reshape, test_y_ts_reshape)
            post_hoc_test_clf_loss_element = torch.reshape(post_hoc_test_clf_loss_element_reshape, (test_y_ts.shape[0], test_y_ts.shape[1]))
            post_hoc_test_clf_loss = torch.mean(post_hoc_test_clf_loss_element)
            _, pred_post_test_pred_indexes = torch.topk(pred_post_hoc_test_y_ts, k = k, dim = -1)
            repeat_gt = torch.argmax(test_y_ts, dim = -1).unsqueeze(-1).repeat(1,1,k)
            post_hoc_test_value, _ = torch.max((pred_post_test_pred_indexes == repeat_gt).int(), dim = -1)
            post_hoc_test_acc = torch.mean(post_hoc_test_value.float())
            post_hoc_test_clf_losses.append(post_hoc_test_clf_loss.item())
            post_hoc_test_accs.append(post_hoc_test_acc.item())
            
            if post_hoc_valid_acc.item() > best_val_acc:
                best_epoch_train_acc = post_hoc_acc.item()
                best_val_acc = post_hoc_valid_acc.item()
                best_epoch_test_acc = post_hoc_test_acc.item()

                best_epoch_pred_post_train_indexes = pred_post_pred_indexes
                best_pred_post_valid_pred_indexes = pred_post_valid_pred_indexes
                best_epoch_pred_post_test_pred_indexes = pred_post_test_pred_indexes

                y_train_true = torch.argmax(train_y_ts, dim = -1).reshape(train_y_ts.shape[0]*train_y_ts.shape[1], -1).squeeze().detach().cpu().numpy() # n
                y_train_pred = best_epoch_pred_post_train_indexes.reshape(best_epoch_pred_post_train_indexes.shape[0]*best_epoch_pred_post_train_indexes.shape[1], -1).squeeze().detach().cpu().numpy()

                y_valid_true = torch.argmax(valid_y_ts, dim = -1).reshape(valid_y_ts.shape[0]*valid_y_ts.shape[1], -1).squeeze().detach().cpu().numpy() # n
                y_valid_pred = best_pred_post_valid_pred_indexes.reshape(pred_post_valid_pred_indexes.shape[0]*pred_post_valid_pred_indexes.shape[1], -1).squeeze().detach().cpu().numpy()

                y_test_true = torch.argmax(test_y_ts, dim = -1).reshape(test_y_ts.shape[0]*test_y_ts.shape[1], -1).squeeze().detach().cpu().numpy() # n
                y_test_pred = best_epoch_pred_post_test_pred_indexes.reshape(pred_post_test_pred_indexes.shape[0]*pred_post_test_pred_indexes.shape[1], -1).squeeze().detach().cpu().numpy()
                
                if figure_vis:
                    conf_matrix = confusion_matrix(y_true = y_valid_true, y_pred = y_valid_pred)
                    fig, ax = plt.subplots(figsize = (len(subclass_labels), len(subclass_labels))) ###L
                    display = ConfusionMatrixDisplay(conf_matrix, display_labels=subclass_labels) ###L
                    display.plot(ax=ax)
                    plt.title(task + ' ' + classify_model_type + ': valid confusion matrix')

                    conf_matrix = confusion_matrix(y_true = y_test_true, y_pred = y_test_pred)
                    fig, ax = plt.subplots(figsize = (len(subclass_labels), len(subclass_labels))) ###L
                    display = ConfusionMatrixDisplay(conf_matrix, display_labels=subclass_labels) ###L
                    display.plot(ax=ax)
                    plt.title(task + ' ' + classify_model_type + ': test confusion matrix')

        outputs = {
            'train_acc': best_epoch_train_acc,
            'valid_acc': best_val_acc,
            'test_acc': best_epoch_test_acc,
            'y_train_true': y_train_true,
            'y_train_pred': y_train_pred,
            'y_valid_true': y_valid_true,
            'y_valid_pred': y_valid_pred,
            'y_test_true': y_test_true,
            'y_test_pred': y_test_pred,
            'train_z': train_z_ts.squeeze().detach().cpu().numpy(),
            'valid_z': valid_z_ts.squeeze().detach().cpu().numpy(),
            'test_z': test_z_ts.squeeze().detach().cpu().numpy()
        }
        
        return outputs
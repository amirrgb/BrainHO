import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import seaborn as sns
from nilearn import image, plotting

RANDOM_SEED = 0


def vis_epoch(model, loader, args, train_data=False, tvt = None):
    model.eval()

    # 用于存储一个 epoch 内所有的预测概率
    all_pred_probs = []
    # 用于存储一个 epoch 内所有的真实标签
    all_true_labels = []

    count = 0
    abm_count = 0
    cor_count = 0

    total_attn_gsg_accum = None
    total_attn_sgn_accum = None
    total_attn_node_accum = None
    total_attn_g2node_accum = None
    total_samples_count = 0

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data_pearson, labels = batch_data
            data_pearson, labels = data_pearson.cuda(), labels.cuda()

            # 判断 labels 是否为 One-Hot (Shape [Batch, 2])
            if labels.dim() > 1 and labels.shape[1] > 1:
                # 假设 Class 1 (index 1) 是 ASD
                asd_indices = (labels[:, 1] == 1).nonzero(as_tuple=True)[0]
            else:
                # 假设 Scalar Labels
                asd_indices = (labels == 1).nonzero(as_tuple=True)[0]

            # 如果当前 Batch 没有 ASD 患者，跳过
            if len(asd_indices) == 0:
                continue

            # 筛选数据
            data_asd = data_pearson[asd_indices]

            pred, attn_node_list, attn_sgn_list, attn_gsg_list, attn_g2node_list = model(data_asd)

            '''----------------------------------------------Metric----------------------------------------------'''
            count+=1
            pred_probs_all_classes = torch.softmax(pred, dim=1)
            y_scores_batch = pred_probs_all_classes[:, 1]
            y_true_batch = labels[:, 1]
            all_pred_probs.append(y_scores_batch.cpu().detach().numpy())
            all_true_labels.append(y_true_batch.cpu().numpy())

            '''----------------------------------------------vis----------------------------------------------'''
            # 取最后一层的 attention: [Batch, Heads, 200, 200]
            # last_layer_attn_gsg = attn_gsg_list[-1]
            # last_layer_attn_sgn = attn_sgn_list[-1]
            # last_layer_attn_node = attn_node_list[-1]
            # last_layer_attn_g2node = attn_g2node_list[-1]

            last_layer_attn_gsg = attn_gsg_list[-2]
            last_layer_attn_sgn = attn_sgn_list[-2]
            last_layer_attn_node = attn_node_list[-2]
            last_layer_attn_g2node = attn_g2node_list[-2]
            # print(f'last_layer_attn_gsg: {last_layer_attn_gsg.shape}')
            # a=[]
            # b=a[1]
            # 在 Heads 维度取平均 [Batch, 200, 200]
            avg_attn_gsg = last_layer_attn_gsg.mean(dim=1)
            avg_attn_sgn = last_layer_attn_sgn.mean(dim=1)
            avg_attn_node = last_layer_attn_node.mean(dim=1)
            avg_attn_g2node = last_layer_attn_g2node.mean(dim=1)

            attn_gsg_sum_batch = avg_attn_gsg.sum(dim=0)  # [200, 200]
            attn_sgn_sum_batch = avg_attn_sgn.sum(dim=0)  # [200, 200]
            attn_node_sum_batch = avg_attn_node.sum(dim=0)  # [200, 200]
            attn_g2node_sum_batch = avg_attn_g2node.sum(dim=0)  # [200, 200]

            if total_attn_gsg_accum is None:
                total_attn_gsg_accum = attn_gsg_sum_batch
            else:
                total_attn_gsg_accum += attn_gsg_sum_batch

            if total_attn_sgn_accum is None:
                total_attn_sgn_accum = attn_sgn_sum_batch
            else:
                total_attn_sgn_accum += attn_sgn_sum_batch

            if total_attn_node_accum is None:
                total_attn_node_accum = attn_node_sum_batch
            else:
                total_attn_node_accum += attn_node_sum_batch

            if total_attn_g2node_accum is None:
                total_attn_g2node_accum = attn_g2node_sum_batch
            else:
                total_attn_g2node_accum += attn_g2node_sum_batch

            total_samples_count += data_asd.size(0)

            # last_layer_attn = torch.stack(attn_node_list).mean(dim=0)


    # AUC-ROC ACC SEN SPEC
    from sklearn.metrics import roc_auc_score, confusion_matrix
    epoch_pred_probs = np.concatenate(all_pred_probs)
    epoch_true_labels = np.concatenate(all_true_labels).astype(int)
    auc = roc_auc_score(epoch_true_labels, epoch_pred_probs)
    threshold = 0.5
    epoch_pred_labels = (epoch_pred_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(epoch_true_labels, epoch_pred_labels).ravel()
    total_samples = tp + tn + fp + fn
    acc = (tp + tn) / total_samples
    sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    if total_samples_count == 0:
        print(f"Warning: Fold {args.cur_fold} - 未发现 ASD 样本 (Check labels)，返回全零矩阵。")
        return np.zeros((200, 200))

    print(f"Fold {args.cur_fold}: 已计算 {total_samples_count} 个 ASD 样本。")
    return auc, acc, sen, spec, total_attn_gsg_accum.cpu().numpy(), total_attn_sgn_accum.cpu().numpy(), total_attn_node_accum.cpu().numpy(), total_attn_g2node_accum.cpu().numpy(), total_samples_count


def vis_learned_composition_yeo7_style(
        avg_attn_sgn,
        mapping_source='/home/path/path/dataset/abide_cc200/node_clus_map.pickle',
        save_name='learned_composition_yeo7_argmax-mdd.png'
):
    import pickle
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    import os

    hex_colors = ['#8D6E63', '#7AA6C2', '#D5896F', '#6A9B78', '#8E7DB3', '#E2B36E', '#6B7A8F', '#A9C59D']
    colors_rgb = [to_rgb(c) for c in hex_colors]

    if not os.path.exists(mapping_source):
        print(f"Error: Mapping file not found {mapping_source}")
        return

    try:
        with open(mapping_source, 'rb') as f:
            node_map = pickle.load(f)

        keys = list(node_map.keys())
        is_1_based_key = (min(keys) == 1)
        num_nodes = 200
        node_network_ids = np.full(num_nodes, -1, dtype=int)

        for k, v in node_map.items():
            idx = k - 1 if is_1_based_key else k
            if 0 <= idx < num_nodes:
                node_network_ids[idx] = v

        valid_mask = node_network_ids != -1
        if node_network_ids[valid_mask].min() == 1:
            node_network_ids[valid_mask] -= 1

    except Exception as e:
        print(f"Error loading mapping: {e}")
        return

    if hasattr(avg_attn_sgn, 'cpu'):
        sgn_matrix = avg_attn_sgn.detach().cpu().numpy()
    elif isinstance(avg_attn_sgn, list):
        sgn_matrix = np.array(avg_attn_sgn)
    else:
        sgn_matrix = avg_attn_sgn

    # shape: (200,)
    node_assignments = np.argmax(sgn_matrix, axis=0)

    sorted_node_indices = np.argsort(node_network_ids)
    sorted_network_ids = node_network_ids[sorted_node_indices]

    image_buffer = np.ones((8, 200, 3))

    for row_idx in range(8):
        for col_idx in range(200):
            real_node_id = sorted_node_indices[col_idx]
            assigned_subgraph = node_assignments[real_node_id]

            if assigned_subgraph == row_idx:
                yeo_id = node_network_ids[real_node_id]
                if 0 <= yeo_id < 8:
                    image_buffer[row_idx, col_idx] = colors_rgb[yeo_id]
            else:
                pass

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.imshow(image_buffer, aspect='auto', interpolation='nearest')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.5)

    boundaries = np.where(sorted_network_ids[:-1] != sorted_network_ids[1:])[0] + 0.5
    for b in boundaries:
        ax.axvline(b, color='black', linestyle='--', linewidth=1, alpha=0.3)

    for i in range(7):
        ax.axhline(i + 0.5, color='black', linestyle='--', linewidth=1, alpha=0.3)

    plt.tight_layout()

    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_name}")

    plt.show()



def export_subgraph_composition_to_csv(
        assignment_matrix,
        mapping_source='/home/path/path/dataset/abide_cc200/node_clus_map.pickle',
        save_path='sg_composition_data.csv'
):
    import pickle
    import numpy as np
    import os

    network_names = ['CB&SB', 'VIS', 'SMN', 'DAN', 'VAN', 'L', 'FPN', 'DMN']

    if not os.path.exists(mapping_source):
        print(f"File missing: {mapping_source}")
        return

    try:
        with open(mapping_source, 'rb') as f:
            node_map = pickle.load(f)
    except Exception as e:
        print(f"Load error: {e}")
        return

    keys = list(node_map.keys())
    is_1_based_key = (min(keys) == 1)
    num_nodes = 200
    gt_labels = np.full(num_nodes, -1, dtype=int)

    for k, v in node_map.items():
        idx = k - 1 if is_1_based_key else k
        if 0 <= idx < num_nodes:
            gt_labels[idx] = v

    valid_mask = gt_labels != -1
    if valid_mask.any() and gt_labels[valid_mask].min() == 1:
        gt_labels[valid_mask] -= 1

    if hasattr(assignment_matrix, 'shape') and len(assignment_matrix.shape) == 2:
        pred_labels = np.argmax(assignment_matrix, axis=0)
    else:
        pred_labels = np.array(assignment_matrix)

    num_learned_sg = 8
    sg_counts = np.zeros((num_learned_sg, 8))

    for node_idx in range(num_nodes):
        sg_id = pred_labels[node_idx]
        yeo_id = gt_labels[node_idx]

        if 0 <= sg_id < num_learned_sg and 0 <= yeo_id < 8:
            sg_counts[sg_id, yeo_id] += 1

    sg_props_scaled = sg_counts / num_nodes

    try:
        with open(save_path, 'w') as f:
            header = "Subgraph," + ",".join(network_names) + "\n"
            f.write(header)

            for i in range(num_learned_sg):
                row_name = f"SG {i + 1}"
                row_vals = ",".join([f"{x:.6f}" for x in sg_props_scaled[i]])
                f.write(f"{row_name},{row_vals}\n")

        print(f"Saved: {save_path}")

    except Exception as e:
        print(f"Save error: {e}")


def export_all_subgraphs_brainnet_clipped_aal(
        avg_attn_sgn,
        avg_attn_node,
        save_dir="BrainNet_Subgraphs_Clipped_AAL",
        nii_path="CC200.nii.gz",
        mapping_source='/home/path/path/dataset/abide_cc200/node_clus_map.pickle',
        aal_mapping_file='/home/path/path/CC200-to-AAL116-Mapping/CC200_to_AAL116_Mapping_with_Names.csv'
):
    import os
    import numpy as np
    import pandas as pd
    import pickle
    from nilearn import image, plotting

    print(f"Generating AAL files: {save_dir}")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if hasattr(avg_attn_sgn, 'cpu'): avg_attn_sgn = avg_attn_sgn.detach().cpu().numpy()
    if hasattr(avg_attn_node, 'cpu'): avg_attn_node = avg_attn_node.detach().cpu().numpy()

    if avg_attn_sgn.shape != (8, 200): avg_attn_sgn = avg_attn_sgn.reshape(8, 200)

    coords = [(0, 0, 0)] * 200
    if os.path.exists(nii_path):
        try:
            img = image.load_img(nii_path)
            extracted = plotting.find_parcellation_cut_coords(img)
            if len(extracted) >= 200:
                coords = extracted[:200]
        except Exception as e:
            print(f"Coord extraction error: {e}")
    else:
        print(f"NIfTI missing: {nii_path}")

    node_colors = np.ones(200, dtype=int)

    if os.path.exists(mapping_source):
        try:
            with open(mapping_source, 'rb') as f:
                node_map = pickle.load(f)

            keys = list(node_map.keys())
            is_1_based_key = (min(keys) == 1)
            vals = list(node_map.values())
            is_1_based_val = (min(vals) == 1)

            for i in range(200):
                query_key = i + 1 if is_1_based_key else i
                if query_key in node_map:
                    raw_val = node_map[query_key]
                    net_id = raw_val - 1 if is_1_based_val else raw_val
                    if 0 <= net_id < 8:
                        node_colors[i] = net_id + 1
        except Exception as e:
            print(f"Map load error: {e}")

    cc200_to_aal = {}
    if os.path.exists(aal_mapping_file):
        try:
            df_map = pd.read_csv(aal_mapping_file)
            for index, row in df_map.iterrows():
                cc_id = int(row['CC200_ID'])
                aal_name = str(row['Mapped_AAL116_Name']).replace(" ", "_")
                cc200_to_aal[cc_id] = aal_name
            print(f"AAL records loaded: {len(cc200_to_aal)}")
        except Exception as e:
            print(f"AAL CSV error: {e}")
    else:
        print(f"AAL mapping missing: {aal_mapping_file}")

    node_assignments = np.argmax(avg_attn_sgn, axis=0)

    for sg_idx in range(8):
        member_indices = np.where(node_assignments == sg_idx)[0]
        num_nodes = len(member_indices)

        if num_nodes == 0:
            print(f"SG {sg_idx + 1}: Empty, skipping")
            continue

        node_filename = os.path.join(save_dir, f"SG_{sg_idx + 1}_AAL.node")

        with open(node_filename, "w") as f:
            for idx in member_indices:
                x, y, z = coords[idx]
                color = node_colors[idx]
                size = avg_attn_sgn[sg_idx, idx]
                cc_id = idx + 1
                aal_name = cc200_to_aal.get(cc_id, "Unknown")
                label = f"{aal_name}"
                f.write(f"{x:.4f}\t{y:.4f}\t{z:.4f}\t{color}\t{size:.6f}\t{label}\n")

        edge_filename = os.path.join(save_dir, f"SG_{sg_idx + 1}_AAL.edge")
        sub_matrix = avg_attn_node[np.ix_(member_indices, member_indices)]
        np.savetxt(edge_filename, sub_matrix, delimiter='\t', fmt='%.6f')

        print(f"SG {sg_idx + 1}: {num_nodes} nodes saved")

    print(f"Done: {os.path.abspath(save_dir)}")






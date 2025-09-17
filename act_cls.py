# python act_cls.py --group_name BusinessEcon --model_name llama_7B_chat --short_model_name r1llama --save_dir ./features
import numpy as np
import argparse
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict


# --- 数据集配置 ---
DATASET_NAME = "cais/mmlu"
DATASET_SPLIT = "test"
CHOICE_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H"]

# --- 推理配置 ---
BATCH_SIZE = 12  # 批处理大小，和原代码保持一致
MAX_TOKENS = 5000






if __name__ == "__main__":
    """
    Command line interface for activation extraction.
    
    Example usage:
        python standalone_activation_extractor.py --model_name llama_7B --dataset_name tqa_mc2
    """
    parser = argparse.ArgumentParser(description="Extract activations from Llama models using PyVene")
    parser.add_argument(
        "--group_name",
        type=str,
        required=True,
        help="The specific group name of the MMLU dataset to process (e.g., 'MathLogic')."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        # default= MODEL_NAME,
        help="The name of the model to use for inference."
    )
    parser.add_argument(
        "--short_model_name",
        type=str,
        required=True,
        # default=SHORT_MODEL_NAME,
        help="The short name of the model for vLLM."
    )
    parser.add_argument('--device', type=int, default=0,
                       help='GPU device ID')
    parser.add_argument('--save_dir', type=str, default='./features',
                       help='Directory to save extracted features')
    
    args = parser.parse_args()


    GROUP_NAME = args.group_name
    MODEL_NAME = args.model_name
    SHORT_MODEL_NAME = args.short_model_name

    print(f"loading features from {args.save_dir}")
    labels = np.load(f'{args.save_dir}/{SHORT_MODEL_NAME}_{GROUP_NAME}_labels.npy')
    layer_wise_activations = np.load(f'{args.save_dir}/{SHORT_MODEL_NAME}_{GROUP_NAME}_layer_wise.npy')
    head_wise_activations = np.load(f'{args.save_dir}/{SHORT_MODEL_NAME}_{GROUP_NAME}_head_wise.npy')

    auc_maxtrix = np.zeros((head_wise_activations.shape[1], head_wise_activations.shape[2]//128))


    print(f"labels shape: {labels.shape}")
    print(f"layer_wise_activations shape: {layer_wise_activations.shape}")
    print(f"head_wise_activations shape: {head_wise_activations.shape}")
    
    # remove the label 2 instance
    mask = labels != 2
    labels = labels[mask]
    layer_wise_activations = layer_wise_activations[mask]
    head_wise_activations = head_wise_activations[mask]

    print(f"labels shape: {labels.shape}")
    print(f"layer_wise_activations shape: {layer_wise_activations.shape}")
    print(f"head_wise_activations shape: {head_wise_activations.shape}")

    head_wise_activations = head_wise_activations.reshape(head_wise_activations.shape[0], head_wise_activations.shape[1], -1, 128)
    head_wise_activations = head_wise_activations.reshape(head_wise_activations.shape[0], -1, 128)
    head_wise_activations = head_wise_activations.transpose(1, 0, 2)

    print(f"head_wise_activations shape: {head_wise_activations.shape}")


    for i in range(head_wise_activations.shape[0]):
        X_feat = head_wise_activations[i]
        clf = LogisticRegression()
        y_prob = cross_val_predict(clf, X_feat, labels, cv=5, method='predict_proba')[:, 1]
        auc = roc_auc_score(labels, y_prob)
        auc_maxtrix[i//len(auc_maxtrix), i%len(auc_maxtrix)] = auc
        # print(f"Feature {i}: AUC = {auc:.4f}")

    # draw the heatmap of auc_maxtrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Sort each row in descending order (larger values first)
    sorted_auc_matrix = np.zeros_like(auc_maxtrix)
    for i in range(auc_maxtrix.shape[0]):
        sorted_indices = np.argsort(auc_maxtrix[i])[::-1]  # Sort in descending order
        sorted_auc_matrix[i] = auc_maxtrix[i][sorted_indices]

    # Create a more polished heatmap
    plt.figure(figsize=(10, 10))
    plt.rcParams['font.size'] = 10

    # Use a better colormap and customize the heatmap
    ax = sns.heatmap(sorted_auc_matrix, 
                     annot=True, 
                     fmt='.3f',  # Show 3 decimal places for better readability
                     cmap='RdYlBu_r',  # Red-Yellow-Blue reversed (red=high, blue=low)
                     cbar_kws={'label': 'AUC Score'},
                     linewidths=0.5,  # Add grid lines between cells
                     square=True)  # Make cells square-shaped

    # Improve the plot appearance
    plt.title(f'AUC Heatmap: {SHORT_MODEL_NAME} - {GROUP_NAME}\n(Rows sorted by AUC values, highest first)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Feature Index (sorted by AUC)', fontsize=14, fontweight='bold')
    plt.ylabel('Layer/Head Group', fontsize=14, fontweight='bold')

    # Rotate x-axis labels if there are many columns
    if sorted_auc_matrix.shape[1] > 10:
        plt.xticks(rotation=45)

    # Improve layout
    plt.tight_layout()

    # Save with higher DPI for better quality
    plt.savefig(f'{args.save_dir}/{SHORT_MODEL_NAME}_{GROUP_NAME}_auc_heatmap.png', 
                dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory

    print(f"Improved heatmap saved to: {args.save_dir}/{SHORT_MODEL_NAME}_{GROUP_NAME}_auc_heatmap.png")
    print(f"Heatmap dimensions: {sorted_auc_matrix.shape[0]} rows × {sorted_auc_matrix.shape[1]} columns")
    print(f"AUC range: {sorted_auc_matrix.min():.3f} - {sorted_auc_matrix.max():.3f}")

    # mask the auc_maxtrix where the value is less than top 5% value
    top_5_percent = np.percentile(auc_maxtrix.reshape(-1), 98)
    auc_maxtrix = np.where(auc_maxtrix >= top_5_percent, 1, 0)
    np.save(f'{args.save_dir}/{SHORT_MODEL_NAME}_{GROUP_NAME}_auc_maxtrix_filtered.npy', auc_maxtrix)
    print(f"AUC matrix filtered, saved to {args.save_dir}/{SHORT_MODEL_NAME}_{GROUP_NAME}_auc_maxtrix_filtered.npy")


    

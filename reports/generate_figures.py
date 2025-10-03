import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for professional plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_training_progress_plots():
    """Create training progress visualizations"""
    
    # Read the results from the best performing model (rear-view dataset)
    results_path = Path("../runs/classify/yolo11n_cls_100e/results.csv")
    
    if results_path.exists():
        df = pd.read_csv(results_path)
        
        # Create training progress plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('YOLO11n Training Progress - Rear-view Dataset', fontsize=16, fontweight='bold')
        
        # Plot 1: Training and Validation Loss
        axes[0, 0].plot(df['epoch'], df['train/loss'], label='Training Loss', linewidth=2, color='#1f77b4')
        axes[0, 0].plot(df['epoch'], df['val/loss'], label='Validation Loss', linewidth=2, color='#ff7f0e')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Top-1 and Top-5 Accuracy
        axes[0, 1].plot(df['epoch'], df['metrics/accuracy_top1'] * 100, 
                       label='Top-1 Accuracy', linewidth=2, color='#2ca02c')
        axes[0, 1].plot(df['epoch'], df['metrics/accuracy_top5'] * 100, 
                       label='Top-5 Accuracy', linewidth=2, color='#d62728')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Model Accuracy Progress')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 100)
        
        # Plot 3: Learning Rate Schedule
        axes[1, 0].plot(df['epoch'], df['lr/pg0'], linewidth=2, color='#9467bd')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')
        
        # Plot 4: Final Performance Summary
        final_metrics = {
            'Top-1 Accuracy': df['metrics/accuracy_top1'].iloc[-1] * 100,
            'Top-5 Accuracy': df['metrics/accuracy_top5'].iloc[-1] * 100,
            'Final Train Loss': df['train/loss'].iloc[-1],
            'Final Val Loss': df['val/loss'].iloc[-1]
        }
        
        metrics_df = pd.DataFrame(list(final_metrics.items()), columns=['Metric', 'Value'])
        bars = axes[1, 1].bar(range(len(metrics_df)), metrics_df['Value'], 
                             color=['#2ca02c', '#d62728', '#1f77b4', '#ff7f0e'])
        axes[1, 1].set_xlabel('Metrics')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].set_title('Final Model Performance')
        axes[1, 1].set_xticks(range(len(metrics_df)))
        axes[1, 1].set_xticklabels(metrics_df['Metric'], rotation=45, ha='right')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('figures/training_progress.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Training progress plots created successfully")
    else:
        print("❌ Training results file not found")

def create_performance_comparison():
    """Create performance comparison between different experiments"""
    
    # Performance data from different experiments
    experiments = {
        'Rear-view Dataset': {
            'accuracy': 95.48,
            'macro_f1': 95.04,
            'weighted_f1': 95.47,
            'classes': 14
        },
        'All-view Dataset': {
            'accuracy': 51.27,
            'macro_f1': 43.61,
            'weighted_f1': 50.08,
            'classes': 15
        }
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Model Performance Comparison Across Datasets', fontsize=16, fontweight='bold')
    
    # Accuracy comparison
    exp_names = list(experiments.keys())
    accuracies = [experiments[exp]['accuracy'] for exp in exp_names]
    
    bars1 = axes[0].bar(exp_names, accuracies, color=['#2ca02c', '#ff7f0e'], alpha=0.8)
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Model Accuracy by Dataset')
    axes[0].set_ylim(0, 100)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # F1-score comparison
    metrics = ['Macro F1', 'Weighted F1']
    rear_f1 = [experiments['Rear-view Dataset']['macro_f1'], 
               experiments['Rear-view Dataset']['weighted_f1']]
    all_f1 = [experiments['All-view Dataset']['macro_f1'], 
              experiments['All-view Dataset']['weighted_f1']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars2 = axes[1].bar(x - width/2, rear_f1, width, label='Rear-view Dataset', 
                       color='#2ca02c', alpha=0.8)
    bars3 = axes[1].bar(x + width/2, all_f1, width, label='All-view Dataset', 
                       color='#ff7f0e', alpha=0.8)
    
    axes[1].set_ylabel('F1-Score (%)')
    axes[1].set_title('F1-Score Comparison')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics)
    axes[1].legend()
    axes[1].set_ylim(0, 100)
    
    # Add value labels
    for bars in [bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('figures/performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Performance comparison plots created successfully")

def create_class_performance_analysis():
    """Create detailed class-wise performance analysis"""
    
    # Class performance data from rear-view dataset (best model)
    class_data = {
        'Brand': ['Audi', 'BYD', 'Kia', 'Lexus', 'Suzuki', 'Benz', 'Ford', 'Toyota', 
                  'Honda', 'BMW', 'Mini', 'Nissan', 'Mazda', 'Hyundai'],
        'F1_Score': [100.0, 100.0, 100.0, 95.65, 95.65, 96.77, 96.30, 95.00, 
                     94.74, 94.73, 93.33, 92.86, 90.91, 84.62],
        'Precision': [100.0, 100.0, 100.0, 91.67, 100.0, 93.75, 100.0, 95.0, 
                      100.0, 96.43, 87.5, 92.86, 90.91, 84.62],
        'Recall': [100.0, 100.0, 100.0, 100.0, 91.67, 100.0, 92.86, 95.0, 
                   90.0, 93.10, 100.0, 92.86, 90.91, 84.62]
    }
    
    df_class = pd.DataFrame(class_data)
    df_class = df_class.sort_values('F1_Score', ascending=True)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    fig.suptitle('Class-wise Performance Analysis - Rear-view Dataset', 
                 fontsize=16, fontweight='bold')
    
    # F1-Score by class
    colors = plt.cm.RdYlGn([f/100 for f in df_class['F1_Score']])
    bars = axes[0].barh(df_class['Brand'], df_class['F1_Score'], color=colors)
    axes[0].set_xlabel('F1-Score (%)')
    axes[0].set_title('F1-Score by Car Make')
    axes[0].set_xlim(0, 105)
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        axes[0].text(width + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{width:.1f}%', ha='left', va='center', fontweight='bold')
    
    # Precision vs Recall scatter plot
    scatter = axes[1].scatter(df_class['Precision'], df_class['Recall'], 
                             c=df_class['F1_Score'], cmap='RdYlGn', 
                             s=100, alpha=0.7, edgecolors='black')
    axes[1].set_xlabel('Precision (%)')
    axes[1].set_ylabel('Recall (%)')
    axes[1].set_title('Precision vs Recall by Car Make')
    axes[1].set_xlim(80, 105)
    axes[1].set_ylim(80, 105)
    axes[1].grid(True, alpha=0.3)
    
    # Add brand labels to scatter points
    for i, brand in enumerate(df_class['Brand']):
        axes[1].annotate(brand, (df_class['Precision'].iloc[i], df_class['Recall'].iloc[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=axes[1])
    cbar.set_label('F1-Score (%)')
    
    # Add diagonal line for perfect precision-recall balance
    axes[1].plot([80, 105], [80, 105], 'k--', alpha=0.3, label='Perfect Balance')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('figures/class_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Class performance analysis plots created successfully")

def create_dataset_overview():
    """Create dataset overview visualization"""
    
    datasets = {
        'Dataset': ['Rear-view\n(yolo_cls_car_makes)', 
                   'Front-Rear\n(yolo_cls_car_makes_front_rear)', 
                   'All-view\n(yolo_cls_car_makes_allview)'],
        'Car_Makes': [14, 14, 15],
        'Viewpoints': ['Rear only', 'Front + Rear', 'All angles'],
        'Complexity': ['Low', 'Medium', 'High'],
        'Best_Accuracy': [95.48, 'Not tested', 51.27]
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Dataset Overview and Characteristics', fontsize=16, fontweight='bold')
    
    # Number of car makes by dataset
    datasets_simple = ['Rear-view', 'Front-Rear', 'All-view']
    car_counts = [14, 14, 15]
    
    bars = axes[0].bar(datasets_simple, car_counts, 
                      color=['#2ca02c', '#ff7f0e', '#d62728'], alpha=0.8)
    axes[0].set_ylabel('Number of Car Makes')
    axes[0].set_title('Car Makes Coverage by Dataset')
    axes[0].set_ylim(0, 16)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # Complexity vs Performance
    complexity_map = {'Low': 1, 'Medium': 2, 'High': 3}
    complexity_scores = [complexity_map[c] for c in ['Low', 'High']]  # Only tested datasets
    accuracies = [95.48, 51.27]
    dataset_labels = ['Rear-view', 'All-view']
    
    scatter = axes[1].scatter(complexity_scores, accuracies, 
                             s=[200, 200], alpha=0.7, 
                             c=['#2ca02c', '#d62728'], edgecolors='black')
    axes[1].set_xlabel('Dataset Complexity')
    axes[1].set_ylabel('Model Accuracy (%)')
    axes[1].set_title('Complexity vs Performance Trade-off')
    axes[1].set_xticks([1, 2, 3])
    axes[1].set_xticklabels(['Low', 'Medium', 'High'])
    axes[1].set_ylim(0, 100)
    axes[1].grid(True, alpha=0.3)
    
    # Add dataset labels
    for i, label in enumerate(dataset_labels):
        axes[1].annotate(label, (complexity_scores[i], accuracies[i]),
                        xytext=(10, 10), textcoords='offset points', 
                        fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('figures/dataset_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Dataset overview plots created successfully")

def main():
    """Main function to generate all visualizations"""
    
    # Create figures directory if it doesn't exist
    Path("figures").mkdir(exist_ok=True)
    
    print("🚀 Starting visualization generation...")
    print("=" * 50)
    
    # Generate all plots
    create_training_progress_plots()
    create_performance_comparison()
    create_class_performance_analysis()
    create_dataset_overview()
    
    print("=" * 50)
    print("🎉 All visualizations generated successfully!")
    print("📁 Check the 'figures/' directory for the generated plots:")
    print("   - training_progress.png")
    print("   - performance_comparison.png") 
    print("   - class_performance_analysis.png")
    print("   - dataset_overview.png")

if __name__ == "__main__":
    main()
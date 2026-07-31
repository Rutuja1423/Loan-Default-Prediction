import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set global visual style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11

def perform_eda(cleaned_data_path, output_img_dir):
    print("=" * 60)
    print("STEP 2: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)
    
    os.makedirs(output_img_dir, exist_ok=True)
    df = pd.read_csv(cleaned_data_path)
    print(f"Loaded {len(df)} records for EDA.")
    
    # Create a string label column for hue-based plots
    df['DefaultLabel'] = df['Default'].map({0: 'Non-Default', 1: 'Default'})
    
    # Palette definition using the string labels
    label_palette = {'Non-Default': '#2ca02c', 'Default': '#d62728'}
    
    # 1. Target Distribution
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    sns.countplot(data=df, x='DefaultLabel', hue='DefaultLabel', palette=label_palette, legend=False, ax=ax[0])
    ax[0].set_title('Loan Default Counts', fontweight='bold')
    ax[0].set_xlabel('Loan Status')
    for p in ax[0].patches:
        ax[0].annotate(f'{int(p.get_height()):,}', (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center', xytext=(0, 8), textcoords='offset points', fontweight='bold')
        
    counts = df['Default'].value_counts().sort_index()
    ax[1].pie(counts, labels=['Non-Default', 'Default'], autopct='%1.1f%%',
              colors=['#2ca02c', '#d62728'], startangle=90, explode=[0, 0.1])
    ax[1].set_title('Default Rate Percentage', fontweight='bold')
    plt.suptitle('Loan Default Target Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "01_target_distribution.png"), dpi=300)
    plt.close()
    print("Saved 01_target_distribution.png")

    # 2. Demographics vs Default
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Age vs Default
    sns.histplot(data=df, x='Age', hue='DefaultLabel', kde=True, palette=label_palette, ax=axes[0, 0], bins=25)
    axes[0, 0].set_title('Age Distribution by Default Status', fontweight='bold')
    
    # Income vs Default
    sns.boxplot(data=df, x='DefaultLabel', y='Income', hue='DefaultLabel', palette=label_palette, legend=False, ax=axes[0, 1])
    axes[0, 1].set_title('Income Distribution by Default Status', fontweight='bold')
    axes[0, 1].set_xlabel('Loan Status')
    
    # Default Rate by Age Group
    df['AgeGroup'] = pd.cut(df['Age'], bins=[17, 30, 45, 60, 100], labels=['18-30', '31-45', '46-60', '60+'])
    age_def = df.groupby('AgeGroup', observed=False)['Default'].mean().reset_index()
    sns.barplot(data=age_def, x='AgeGroup', y='Default', color='#3182bd', ax=axes[1, 0])
    axes[1, 0].set_title('Default Rate by Age Group', fontweight='bold')
    axes[1, 0].set_ylabel('Default Rate')
    for p in axes[1, 0].patches:
        axes[1, 0].annotate(f'{p.get_height():.1%}', (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', xytext=(0, 8), textcoords='offset points')

    # Default Rate by Income Bracket
    df['IncomeBracket'] = pd.qcut(df['Income'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    inc_def = df.groupby('IncomeBracket', observed=False)['Default'].mean().reset_index()
    sns.barplot(data=inc_def, x='IncomeBracket', y='Default', color='#3182bd', ax=axes[1, 1])
    axes[1, 1].set_title('Default Rate by Income Quintile', fontweight='bold')
    axes[1, 1].set_ylabel('Default Rate')
    for p in axes[1, 1].patches:
        axes[1, 1].annotate(f'{p.get_height():.1%}', (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', xytext=(0, 8), textcoords='offset points')

    plt.suptitle('Demographics & Default Risk Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "02_demographics_default.png"), dpi=300)
    plt.close()
    print("Saved 02_demographics_default.png")

    # 3. Financial Risk Features
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Credit Score
    sns.boxplot(data=df, x='DefaultLabel', y='CreditScore', hue='DefaultLabel', palette=label_palette, legend=False, ax=axes[0, 0])
    axes[0, 0].set_title('Credit Score by Default Status', fontweight='bold')
    axes[0, 0].set_xlabel('Loan Status')
    
    # Interest Rate
    sns.boxplot(data=df, x='DefaultLabel', y='InterestRate', hue='DefaultLabel', palette=label_palette, legend=False, ax=axes[0, 1])
    axes[0, 1].set_title('Interest Rate by Default Status', fontweight='bold')
    axes[0, 1].set_xlabel('Loan Status')
    
    # DTI Ratio
    sns.histplot(data=df, x='DTIRatio', hue='DefaultLabel', kde=True, palette=label_palette, ax=axes[1, 0], bins=25)
    axes[1, 0].set_title('DTI Ratio Distribution by Default Status', fontweight='bold')
    
    # Months Employed
    sns.boxplot(data=df, x='DefaultLabel', y='MonthsEmployed', hue='DefaultLabel', palette=label_palette, legend=False, ax=axes[1, 1])
    axes[1, 1].set_title('Months Employed by Default Status', fontweight='bold')
    axes[1, 1].set_xlabel('Loan Status')
    
    plt.suptitle('Financial Profile & Risk Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "03_financial_metrics.png"), dpi=300)
    plt.close()
    print("Saved 03_financial_metrics.png")

    # 4. Categorical Breakdown
    cat_cols = ['Education', 'EmploymentType', 'LoanPurpose', 'MaritalStatus', 'HasCoSigner', 'HasMortgage']
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, col in enumerate(cat_cols):
        cat_def = df.groupby(col)['Default'].mean().reset_index().sort_values(by='Default', ascending=False)
        sns.barplot(data=cat_def, x=col, y='Default', palette='Blues_r', hue=col, legend=False, ax=axes[idx])
        axes[idx].set_title(f'Default Rate by {col}', fontweight='bold')
        axes[idx].set_ylabel('Default Rate')
        axes[idx].tick_params(axis='x', rotation=30)
        for p in axes[idx].patches:
            axes[idx].annotate(f'{p.get_height():.1%}', (p.get_x() + p.get_width() / 2., p.get_height()),
                               ha='center', va='center', xytext=(0, 8), textcoords='offset points')
            
    plt.suptitle('Categorical Factors Influencing Loan Default', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "04_categorical_defaults.png"), dpi=300)
    plt.close()
    print("Saved 04_categorical_defaults.png")

    # 5. Correlation Heatmap
    num_df = df.select_dtypes(include=[np.number])
    corr = num_df.corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
    plt.title('Numerical Features Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_img_dir, "05_correlation_heatmap.png"), dpi=300)
    plt.close()
    print("Saved 05_correlation_heatmap.png")

    print("\nEDA completed! Visualizations saved in:", output_img_dir)

if __name__ == "__main__":
    clean_path = os.path.join("data", "cleaned", "loan_data_cleaned.csv")
    img_dir = "images"
    perform_eda(clean_path, img_dir)

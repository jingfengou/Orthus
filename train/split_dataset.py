import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import os

def split_data(parquet_path, output_dir, test_size=0.2):
    """
    加载 Parquet 数据集, 使用 pandas 和 scikit-learn 进行分层抽样, 
    并将结果保存为 train.jsonl 和 test.jsonl。
    """
    print(f"Loading dataset from: {parquet_path} using pandas...")
    
    # --- 修改点 1: 使用 pandas 直接加载 Parquet 文件 ---
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"Error loading Parquet file with pandas: {e}")
        return
            
    print(f"Total records found: {len(df)}")
    
    # 检查 'Task' 列是否存在
    if 'Task' not in df.columns:
        raise ValueError("The input file must contain a 'Task' column for stratified splitting.")
        
    print(f"Performing stratified split based on 'Task' column (test_size={test_size})...")

    # --- 修改点 2: 使用 scikit-learn 的 train_test_split 进行分层抽样 ---
    # 这个函数非常强大，可以对任何类型的列进行分层
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=42,      # 确保每次划分结果都一样
        stratify=df['Task']   # 关键：根据 'Task' 列进行分层
    )
    
    print(f"Training set size: {len(train_df)}")
    print(f"Testing set size: {len(test_df)}")
    
    # 3. 将划分好的数据集保存为 .jsonl 文件
    train_file_path = os.path.join(output_dir, "train.jsonl")
    test_file_path = os.path.join(output_dir, "test.jsonl")
    
    train_df.to_json(train_file_path, orient='records', lines=True)
    test_df.to_json(test_file_path, orient='records', lines=True)
    
    print(f"Training data saved to: {train_file_path}")
    print(f"Testing data saved to: {test_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split Parquet dataset into train/test JSONL files.")
    parser.add_argument("--parquet_path", type=str, required=True, 
                        help="Path to the input test-00000-of-00001.parquet file.")
    parser.add_argument("--output_dir", type=str, default=".", 
                        help="Directory to save the train.jsonl and test.jsonl files.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    split_data(args.parquet_path, args.output_dir)
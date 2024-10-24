def check_batch_indices(filename='batch_dataset_indices.txt'):
    with open(filename, 'r') as f:
        for line in f:
            # 提取 batch 的 dataset indices 部分
            if "Batch" in line:
                indices = line.split("Dataset Indices: ")[-1].strip().strip("[]")
                indices_list = [int(idx) for idx in indices.split(", ")]
                
                # 检查是否所有的 indices 都相同
                if len(set(indices_list)) > 1:
                    print(f"Inconsistent indices found: {line.strip()}")

check_batch_indices()
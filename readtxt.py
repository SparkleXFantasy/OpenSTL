import ast

def parse_batch_tensors_shape_log(filename='/home/bingxing2/ailab/suencheng/encheng/code/OpenSTL/batch_structure_log.txt'):
    """
    读取并解析指定路径的日志文件中的信息，以提取每个 batch 中 tensor1 和 tensor2 的形状。
    
    Args:
        filename (str): 要读取的日志文件的完整路径。
    """
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"文件未找到，请确认路径是否正确: {filename}")
        return

    batch_idx = None
    for line in lines:
        line = line.strip()
        if line.startswith("Batch"):
            # 读取 Batch 索引
            batch_idx = line
            print(f"\n{batch_idx}")
        elif line.startswith("Sample"):
            # 读取 Sample 信息
            try:
                # 找到每个 sample 的值部分
                value_start = line.find("Value: ") + len("Value: ")
                value_str = line[value_start:]
                
                # 使用 ast.literal_eval 来安全地解析列表中的值
                value_list = ast.literal_eval(value_str)
                if isinstance(value_list, list) and len(value_list) > 0 and isinstance(value_list[0], tuple):
                    _, (tensor1, tensor2) = value_list[0]
                    # 打印 tensor1 和 tensor2 的形状
                    print(f"  Sample tensor1 shape: {tensor1.shape}, tensor2 shape: {tensor2.shape}")
                else:
                    print("  解析的值不符合预期结构，无法提取张量形状。")
            except (ValueError, SyntaxError) as e:
                print(f"  无法解析样本值，发生错误: {e}")

# 调用函数，指定文本文件的路径
parse_batch_tensors_shape_log('/home/bingxing2/ailab/suencheng/encheng/code/OpenSTL/batch_structure_log.txt')

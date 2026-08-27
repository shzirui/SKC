import json
import random

def sample_json_data(input_file, output_file, sample_rate=0.25):
    # 1. 读取文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    sampled_data = {}
    
    # 2. 遍历每个类别并进行采样
    for category, items in data.items():
        # 计算采样数量（向上取整，确保每个类别至少有一个，除非原列表为空）
        sample_size = max(1, int(len(items) * sample_rate)) if len(items) > 0 else 0
        
        # 随机采样
        sampled_data[category] = random.sample(items, sample_size)
        
        print(f"类别: {category:<20} | 原总数: {len(items):<3} | 采样数: {sample_size}")

    # 3. 写入新文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n采样完成！结果已保存至: {output_file}")

# 执行
if __name__ == "__main__":
    # 替换成你的文件名
    sample_json_data('trainset_new_181.json', 'trainset_replay.json')
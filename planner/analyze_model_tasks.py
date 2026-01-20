import json
import os
import sys
from pathlib import Path

def analyze_task_allocation(json_file_path, threshold=4):
    """
    分析任务分配情况
    
    参数:
        json_file_path: dataset_results.json 文件的路径
        threshold: 难度阈值，默认为4
    
    返回:
        包含统计信息的字典
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取JSON文件出错: {e}")
        return None
    
    # 初始化计数器
    total_tasks = 0
    small_model_tasks = 0
    large_model_tasks = 0
    
    # 遍历所有问题
    for problem in data:
        tasks = problem.get('tasks', {})
        
        # 遍历每个问题的任务
        for task_id, task_info in tasks.items():
            total_tasks += 1
            
            # 获取任务难度（如果不存在，假设为0）
            difficulty = int(task_info.get('Difficulty', 0))
            
            # 根据难度阈值判断使用哪个模型
            if difficulty < threshold:
                small_model_tasks += 1
            else:
                large_model_tasks += 1
    
    # 计算百分比
    small_model_percentage = (small_model_tasks / total_tasks * 100) if total_tasks > 0 else 0
    large_model_percentage = (large_model_tasks / total_tasks * 100) if total_tasks > 0 else 0
    
    return {
        'total_tasks': total_tasks,
        'small_model_tasks': small_model_tasks,
        'large_model_tasks': large_model_tasks,
        'small_model_percentage': small_model_percentage,
        'large_model_percentage': large_model_percentage
    }

def generate_markdown_section(stats):
    """
    生成Markdown格式的任务分配统计部分
    
    参数:
        stats: 统计信息字典
    
    返回:
        Markdown格式的文本
    """
    markdown = "## 任务分配统计\n\n"
    markdown += f"- 总任务数: {stats['total_tasks']}\n"
    markdown += f"- 小模型执行任务数: {stats['small_model_tasks']}\n"
    markdown += f"- 大模型执行任务数: {stats['large_model_tasks']}\n"
    markdown += f"- 小模型任务占比: {stats['small_model_percentage']:.2f}%\n"
    markdown += f"- 大模型任务占比: {stats['large_model_percentage']:.2f}%\n"
    
    return markdown

def main():

    json_file_path = r"data_reports/dataset_results.json"
    threshold = 5
    
    # 确保文件存在
    if not os.path.exists(json_file_path):
        print(f"错误: 文件 {json_file_path} 不存在")
        return
    
    # 分析任务分配
    stats = analyze_task_allocation(json_file_path, threshold)
    if not stats:
        return
    
    # 生成并打印Markdown部分
    markdown_section = generate_markdown_section(stats)
    print(markdown_section)
    
    # 找到对应的报告文件
    json_dir = Path(json_file_path).parent
    report_path = json_dir / "dataset_report.md"
    
    if os.path.exists(report_path):
        print(f"找到对应的报告文件: {report_path}")
        print("您可以将上面的统计结果添加到此报告文件中。")
    else:
        print("未找到对应的报告文件。")

if __name__ == "__main__":
    main()

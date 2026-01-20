# Update 7/10/2025 from https://openrouter.ai/models

from typing import Dict, Any, List, Tuple
import transformers
import os
import sys
from log_config import setup_logger, get_logger, log_separator

# 初始化tokenizer（全局变量）
try:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "deepseek_v3_tokenizer"),
        trust_remote_code=True
    )
except Exception as e:
    print(f"警告: 无法加载DeepSeek tokenizer: {e}")
    logger = get_logger()
    logger.error(f"无法加载DeepSeek tokenizer: {e}", exc_info=True)
    tokenizer = None

def count_tokens(text):
    """使用DeepSeek tokenizer计算文本的token数量
    
    参数:
        text: 需要计算token数量的文本
    
    返回:
        token数量 (整数)
    """
    if tokenizer:
        try:
            tokens = tokenizer.encode(text)
            return len(tokens)
        except Exception as e:
            print(f"警告: 使用DeepSeek tokenizer计算token失败: {e}")
            logger = get_logger()
            logger.error(f"使用DeepSeek tokenizer计算token失败: {e}", exc_info=True)

    # 回退方法：使用简单的估计方法 (4个字符≈1个token)
    return len(text) // 4

def calculate_average(values):
    if not values:
        return 0
    return sum(values) / len(values)

def get_model_performance(model_name: str) -> Dict[str, float]:
    """
    根据模型名称返回其性能指标（延迟和吞吐量）
    
    参数:
        model_name: 模型名称（字符串或包含模型名称的字符串）
    
    返回:
        包含输入和输出性能指标的字典 {'latency': float, 'throughput': float}
    """
    # 标准化模型名称为小写以便匹配
    model_name_lower = model_name.lower()
    
    # Claude模型性能
    if "claude-3-5" in model_name_lower or "claude-3.5" in model_name_lower or "claude-3-5-sonnet" in model_name_lower:
        return {"latency": calculate_average([0.76, 1.41, 1.75, 1.54, 1.23]), "throughput": calculate_average([52.59, 49.44, 56.96, 45.58, 52.89])}
    elif "claude-3-7" in model_name_lower or "claude-3.7" in model_name_lower or "claude-3-7-sonnet" in model_name_lower:
        return {"latency": calculate_average([1.41, 3.99, 2.74, 2.40]), "throughput": calculate_average([56.31, 69.85, 53.88, 90.04])}

    # Gemini模型性能
    elif "gemini-2.5-flash-thinking" in model_name_lower:
        return {"latency": calculate_average([0.41, 0.59, 1.21]), "throughput": calculate_average([106.6, 118.1, 86.42])}
    elif "gemini-2.5-pro" in model_name_lower:
        return {"latency": calculate_average([2.20, 2.99, 2.34]), "throughput": calculate_average([100.7, 94.10, 86.46])}

    # OpenAI模型性能
    elif "gpt-4o" in model_name_lower:
        return {"latency": calculate_average([0.50, 0.97]), "throughput": calculate_average([110.5, 178.5])}
    elif "gpt-5" in model_name_lower:
        return {"latency": calculate_average([6.96, 7.03, 5.23]), "throughput": calculate_average([53.20, 53.43, 45.09])}
    elif "gpt-4.1-mini" in model_name_lower:
        return {"latency": 0.70, "throughput": 69.59}
    elif "gpt-4.1" in model_name_lower:
        return {"latency": 0.96, "throughput": 50.59}

    # DeepSeek系列
    elif "deepseek-r1" in model_name_lower:
        return {"latency": calculate_average([2.37, 1.18, 0.88, 0.97, 0.51]), 
                "throughput": calculate_average([44.26, 48.32, 65.32, 35.83, 38.71])}
    elif "deepseek-chat" in model_name_lower:
        return {"latency": calculate_average([1.89, 0.91, 3.13, 0.67, 1.40]), 
                "throughput": calculate_average([36.14, 54.12, 20.80, 24.61, 24.18])}
    elif "deepseek-reasoner" in model_name_lower:
        return {"latency": calculate_average([2.37, 1.18, 0.88, 0.97, 0.51]), 
                "throughput": calculate_average([44.26, 48.32, 65.32, 35.83, 38.71])}

    # Grok系列
    elif "grok-4" in model_name_lower:
        return {"latency": 12.65, "throughput": 36.37}

    # Llama系列
    elif "llama3-8b" in model_name_lower or "llama-3-8b" in model_name_lower:
        return {"latency": calculate_average([0.55, 0.69, 0.86, 0.48]), 
                "throughput": calculate_average([89.21, 94.59, 148.4, 15.58])}
    elif "llama-3.2-3b" in model_name_lower or "llama3.2-3b" in model_name_lower:
        return {"latency": calculate_average([0.54, 0.40, 0.76, 0.38, 0.37]), "throughput": calculate_average([128.3, 53.67, 181.8, 229.9, 96.15])}
    elif "llama-3.3-70b" in model_name_lower or "llama3.3-70b" in model_name_lower:
        return {"latency": calculate_average([0.55, 0.25, 0.72, 0.68, 0.57]), 
                "throughput": calculate_average([57.97, 40.28, 51.03, 75.79, 41.67])}
    elif "llama-3.2-1b" in model_name_lower or "llama3.2-1b" in model_name_lower:
        return {"latency": calculate_average([1.45, 0.48, 0.31]), "throughput": calculate_average([17.43, 61.09, 439.1])}
    
    # Qwen系列
    elif "qwen3-4b" in model_name_lower:
        return {"latency": 0.69, "throughput": 184.1}
    elif "qwen2.5-3b" in model_name_lower:
        return {"latency": 0.69, "throughput": 64.53}
    elif "qwen3-235b-a22b" in model_name_lower:
        return {"latency": calculate_average([0.31, 1.34]), "throughput":calculate_average([52.82, 88.24])}
    elif "qwen3-1.7b" in model_name_lower:
        return {"latency": 0.69, "throughput": 184.1}
    elif "qwen3-0.6b" in model_name_lower:
            return {"latency": 0.69, "throughput": 184.1}

    # 本地模型
    elif "saves" in model_name_lower:
        return {"latency": 0.5, "throughput": 71.2}

    else:
        print(f"警告: 未识别的模型 '{model_name}'，使用默认性能")
        logger = get_logger()
        logger.warning(f"未识别的模型 '{model_name}'，使用默认性能: latency:0.5s, throughput:71.2tokens/s。请在 output_performance.py 中添加此模型的性能信息。")
        return {"latency": 0.5, "throughput": 71.2}

def calculate_theoretical_time(model_name: str, tokens: int) -> Dict[str, float]:
    """
    根据模型性能指标和token数量计算理论时间
    
    参数:
        model_name: 模型名称
        tokens: 生成的token数量
    
    返回:
        包含各项时间指标的字典 {'latency': float, 'generation_time': float, 'total_time': float}
    """
    performance = get_model_performance(model_name)
    
    # 延迟时间（TTFT - Time To First Token）
    latency = performance["latency"]
    
    # 生成时间（根据吞吐量计算）
    generation_time = tokens / performance["throughput"] if tokens > 0 else 0
    
    # 总时间 = 延迟 + 生成时间
    total_time = latency + generation_time
    
    return {
        "latency": latency,
        "generation_time": generation_time,
        "total_time": total_time
    }

def generate_theoretical_performance_report(tasks, config, planner_output=None):
    """
    生成基于理论性能指标的报告
    
    参数:
        tasks: 任务字典
        config: 模型配置对象
        planner_output: planner实际输出的规划结果，包含token数量等信息，默认为None
    
    返回:
        理论性能报告文本
    """
    # 收集任务执行情况
    small_model_tasks = []
    large_model_tasks = []
    
    # 按执行顺序整理任务
    sorted_tasks = sorted(tasks.items(), key=lambda x: int(x[0]))
    
    # 遍历每个任务，根据难度分配到对应模型
    for step_id, task in sorted_tasks:
        # 提取token数量，默认为1000
        token_str = task.get('Token', '1000')
        try:
            tokens = int(token_str)
        except ValueError:
            tokens = 1000  # 默认值
            
        # 提取难度，根据难度选择模型
        difficulty = task.get('Difficulty', '0')
        
        # 创建任务信息
        task_info = {
            'step_id': step_id,
            'task': task.get('Task', f'步骤 {step_id}'),
            'tokens': tokens,
            'rely': task.get('Rely', '').split(',') if task.get('Rely', '') else []
        }
        
        # 根据难度判断使用哪个模型
        if int(difficulty) >= config.threshold:
            # 大模型任务
            large_model_tasks.append(task_info)
        else:
            # 小模型任务
            small_model_tasks.append(task_info)
    
    # 构建完整的任务依赖图
    dependency_graph = {}
    for step_id, task in sorted_tasks:
        rely_str = task.get('Rely', '')
        dependencies = [dep for dep in rely_str.split(',') if dep]  # 过滤空依赖
        dependency_graph[step_id] = dependencies
    
    # 获取模型名称
    small_model_name = config.small_model
    large_model_name = config.large_model
    router_model_name = config.router_model if not config.use_local_router else config.local_router_model
    
    # 计算规划阶段（Planner）的理论时间
    router_performance = get_model_performance(router_model_name)
    planner_latency = router_performance['latency']
    
    # 使用实际的planner输出token数量计算生成时间
    if planner_output and isinstance(planner_output, dict):
        # 如果提供了planner输出信息，使用实际的token计数
        plan_tokens = planner_output.get('completion_tokens', 0)
    else:
        # 如果没有提供planner输出信息，基于任务数估算
        plan_tokens = len(tasks) * 100  # 每个任务约需100个token
        print(f"使用估算的planner输出token数: {plan_tokens}")
        logger = get_logger()
        logger.warning(f"使用估算的planner输出token数: {plan_tokens}")
    
    # 计算路由模型的理论时间（初始化 + 生成计划）
    planner_generation_time = plan_tokens / router_performance['throughput']
    planner_total_time = planner_latency + planner_generation_time
    
    # 模拟任务执行的理论时间，考虑依赖关系和并行执行
    # 每个任务的开始时间和结束时间
    earliest_start_times = {}
    earliest_finish_times = {}
    
    # 任务的执行时间映射
    task_execution_times = {}
    for step_id, task in sorted_tasks:
        token_str = task.get('Token', '1000')
        try:
            tokens = int(token_str)
        except ValueError:
            tokens = 1000
            
        difficulty = task.get('Difficulty', '0')
        
        # 根据难度选择模型
        if int(difficulty) >= config.threshold:
            model_name = large_model_name
        else:
            model_name = small_model_name
            
        # 计算任务执行时间
        time_data = calculate_theoretical_time(model_name, tokens)
        task_execution_times[step_id] = time_data['total_time']
    
    # 计算任务的最早开始和完成时间，模拟真实调度过程
    max_workers = 10  # 默认并行工作线程数，可以从config中获取

    # 模拟任务执行过程
    simulation_result, planner_time = simulate_task_execution(
        sorted_tasks, 
        dependency_graph, 
        task_execution_times, 
        max_workers, 
        planner_total_time,
        config
    )
    
    total_execution_time = simulation_result['total_time']
    task_timelines = simulation_result['task_timelines']
    worker_allocation = simulation_result['worker_allocation']
    task_planning_times = simulation_result['task_planning_times']
    
    # 计算各模型的理论时间总和（不考虑并行）
    small_model_theoretical_time = sum(
        task_execution_times[task['step_id']] for task in small_model_tasks
    )
    
    large_model_theoretical_time = sum(
        task_execution_times[task['step_id']] for task in large_model_tasks
    )
    
    # 构建报告
    report = "# 理论性能模型分析\n\n"
    
    # 添加模型性能参数
    small_perf = get_model_performance(small_model_name)
    large_perf = get_model_performance(large_model_name)
    router_perf = get_model_performance(router_model_name)
    
    report += "## 模型性能参数\n\n"
    report += "| 模型 | 延迟 (秒) | 吞吐量 (tokens/s) |\n"
    report += "| --- | --- | --- |\n"
    report += f"| 小模型 ({small_model_name}) | {small_perf['latency']:.3f} | {small_perf['throughput']:.2f} |\n"
    report += f"| 大模型 ({large_model_name}) | {large_perf['latency']:.3f} | {large_perf['throughput']:.2f} |\n"
    report += f"| 路由模型 ({router_model_name}) | {router_perf['latency']:.3f} | {router_perf['throughput']:.2f} |\n\n"
    
    # 添加执行流程理论时间
    report += "## 执行流程理论时间\n\n"
    report += "| 阶段 | 理论时间 (秒) | 百分比 |\n"
    report += "| --- | --- | --- |\n"
    report += f"| 规划阶段总时间 (Planner) | {planner_time:.3f} | 100% |\n"
    
    # 计算有多少任务在规划阶段就开始执行
    tasks_started_during_planning = sum(1 for task_id, timeline in task_timelines.items() if timeline['start_time'] < planner_time)
    tasks_percentage = (tasks_started_during_planning / len(task_timelines)) * 100 if task_timelines else 0
    
    # 最后一个任务规划的时间
    last_planning_time = max(task_planning_times.values()) if task_planning_times else 0
    
    # 最后一个任务完成的时间
    last_task_completion_time = max(timeline['end_time'] for timeline in task_timelines.values()) if task_timelines else 0
    
    # 计算规划与执行的重叠时间（从第一个任务规划完成到最后一个任务规划完成这段时间内有多少任务在执行）
    first_task_planning_time = min(task_planning_times.values()) if task_planning_times else 0
    tasks_executing_during_planning = sum(1 for task_id, timeline in task_timelines.items() if timeline['start_time'] < last_planning_time)
    
    # 规划阶段执行的任务比例
    planning_execution_ratio = (tasks_executing_during_planning / len(task_timelines)) * 100 if task_timelines else 0
    
    # 理论并行效率（任务总执行时间 / 实际总时间）
    total_task_time = sum(task_execution_times.values())
    parallel_efficiency = (total_task_time / last_task_completion_time) * 100 if last_task_completion_time > 0 else 0
    
    # 流水线效率（规划与执行重叠时的加速）
    if planner_total_time + total_task_time > 0 and last_task_completion_time > 0:
        pipeline_speedup = (planner_total_time + total_task_time) / last_task_completion_time
    else:
        pipeline_speedup = 1.0
    
    report += f"| 规划过程中启动的任务数 | {tasks_started_during_planning} / {len(task_timelines)} | {tasks_percentage:.1f}% |\n"
    report += f"| 规划与执行重叠的任务数 | {tasks_executing_during_planning} / {len(task_timelines)} | {planning_execution_ratio:.1f}% |\n"
    report += f"| 第一个任务规划完成时间 | {first_task_planning_time:.3f} | - |\n"
    report += f"| 最后一个任务规划完成时间 | {last_planning_time:.3f} | - |\n"
    report += f"| 最后一个任务执行完成时间 | {last_task_completion_time:.3f} | - |\n"
    report += f"| 任务总执行时间(累计) | {total_task_time:.3f} | - |\n"
    report += f"| 流水线加速比 | {pipeline_speedup:.2f}x | - |\n"
    report += f"| 并行效率 | {parallel_efficiency:.1f}% | - |\n\n"
    
    # 添加任务类型理论时间
    report += "## 任务类型理论时间\n\n"
    report += "| 模型类型 | 任务数 | 顺序执行时间 (秒) | 并行加速比 |\n"
    report += "| --- | --- | --- | --- |\n"
    
    # 计算并行加速比
    sequential_time = planner_total_time + small_model_theoretical_time + large_model_theoretical_time
    parallel_speedup = sequential_time / total_execution_time if total_execution_time > 0 else 1.0
    
    report += f"| 小模型任务 | {len(small_model_tasks)} | {small_model_theoretical_time:.3f} | - |\n"
    report += f"| 大模型任务 | {len(large_model_tasks)} | {large_model_theoretical_time:.3f} | - |\n"
    report += f"| 规划模型 | 1 | {planner_total_time:.3f} | - |\n"
    report += f"| 顺序总时间 | - | {sequential_time:.3f} | - |\n"
    report += f"| 并行总时间 | - | {total_execution_time:.3f} | {parallel_speedup:.2f}x |\n\n"
    
    # 添加任务明细
    report += "## 任务执行明细\n\n"
    report += "| 步骤ID | 任务描述 | 使用模型 | 理论开始时间 (秒) | 理论结束时间 (秒) | 理论执行时间 (秒) | 工作线程 |\n"
    report += "| --- | --- | --- | --- | --- | --- | --- |\n"
    
    for step_id, task in sorted_tasks:
        task_desc = task.get('Task', f'步骤 {step_id}')
        difficulty = task.get('Difficulty', '0')
        
        # 根据难度判断使用哪个模型
        if int(difficulty) >= config.threshold:
            model_type = "大模型"
        else:
            model_type = "小模型"
            
        start_time = task_timelines[step_id]['start_time']
        end_time = task_timelines[step_id]['end_time']
        duration = end_time - start_time
        worker_id = worker_allocation.get(step_id, "N/A")
        
        report += f"| {step_id} | {task_desc} | {model_type} | {start_time:.3f} | {end_time:.3f} | {duration:.3f} | {worker_id} |\n"
    
    # 添加理论执行甘特图描述
    report += "\n## 理论执行甘特图\n\n"
    report += "```\n"
    report += generate_gantt_chart(task_timelines, max_workers)
    report += "```\n\n"
        
    return report

def simulate_task_execution(sorted_tasks, dependency_graph, task_execution_times, max_workers, planner_time, config=None):
    """
    模拟并行任务执行，考虑依赖关系和工作线程数限制
    
    参数:
        sorted_tasks: 按ID排序的任务
        dependency_graph: 任务依赖关系图
        task_execution_times: 每个任务的执行时间
        max_workers: 最大并行工作线程数
        planner_time: 规划阶段的总时间（包含所有任务的规划）
        config: 模型配置对象，用于获取路由模型名称
        
    返回:
        包含总执行时间和任务时间线的字典
    """
    # 检查是否提供了配置对象
    if config is None:
        # 如果没有提供配置，导入默认配置
        import config as config_module
        config = config_module.Config()
    
    # 记录每个任务的开始时间和结束时间
    task_timelines = {}
    
    # 工作线程分配情况
    worker_allocation = {}  # step_id -> worker_id
    
    # 已完成任务
    completed_tasks = {}  # step_id -> 完成时间
    
    # 获取路由模型名称和性能参数
    router_model_name = config.router_model if hasattr(config, 'router_model') and not (hasattr(config, 'use_local_router') and config.use_local_router) else config.local_router_model if hasattr(config, 'local_router_model') else "claude-3-5-sonnet-latest"
    router_performance = get_model_performance(router_model_name)
    router_latency = router_performance["latency"]
    router_throughput = router_performance["throughput"]
    
    # 第一个任务的规划时间需要加上初始延迟
    cumulative_time = router_latency
    step_xml = '<Plan>'
    step_tokens = count_tokens(step_xml)
    cumulative_time += step_tokens / router_throughput
    # 存储每个任务的plan输出时间点
    task_available_time = {}
    
    # 按照任务ID顺序重构每个步骤的XML内容并计算规划输出时间点
    for i, (step_id, task_data) in enumerate(sorted_tasks):
        # 重构每个任务的XML格式
        task_content = task_data.get('Task', f'步骤 {step_id}')
        difficulty = task_data.get('Difficulty', '1')
        token_str = task_data.get('Token', '30')
        rely = task_data.get('Rely', '')
        
        # 重构Step标签 - 格式如: <Step ID="1" Task="..." Difficulty="2" Token="25" Rely=""/>
        step_xml = f'<Step ID="{step_id}" Task="{task_content}" Difficulty="{difficulty}" Token="{token_str}" Rely="{rely}"/>'
        # 使用DeepSeek tokenizer估算该XML内容的token数量
        step_tokens = count_tokens(step_xml)
        
        # 计算该步骤规划所需的时间
        step_planning_time = step_tokens / router_throughput
        
        # 累加规划时间（从延迟开始）
        cumulative_time += step_planning_time
        task_available_time[step_id] = cumulative_time
    step_xml = '</Plan>'
    step_tokens = count_tokens(step_xml)
    cumulative_time += step_tokens / router_throughput
    planner_time = cumulative_time
    # 按照新算法逻辑实现任务执行时间计算
    # 核心思想：一旦任务被规划出来且其依赖任务完成，就可以立即开始执行
    # 依次处理每个任务，计算其开始时间和结束时间
    for step_id, task_data in sorted_tasks:
        # 获取任务的依赖关系
        dependencies = dependency_graph.get(step_id, [])
        
        # 计算任务最早可以开始的时间
        # 1. 任务被规划出来的时间点 - 这是任务可执行的前提条件
        plan_time = task_available_time[step_id]
        
        # 2. 所有依赖任务的完成时间 - 只有依赖任务都完成，才能开始执行当前任务
        dependency_finish_time = 0
        if dependencies and dependencies != ['']: 
            dep_times = [completed_tasks.get(dep, 0) for dep in dependencies if dep]
            if dep_times:  # 如果有实际依赖
                dependency_finish_time = max(dep_times)
        
        # 任务开始时间 = max(任务计划完成时间, 所有依赖任务的完成时间)
        # 这体现了"流式执行"的特点：一旦任务被规划出来且依赖满足，就可以立即开始执行
        start_time = max(plan_time, dependency_finish_time)
        
        # 计算任务执行时间（基于token数和模型性能）
        execution_time = task_execution_times[step_id]
        
        # 任务结束时间 = 开始时间 + 执行时间
        end_time = start_time + execution_time
        
        # 记录任务时间线，包括开始和结束时间
        task_timelines[step_id] = {
            "start_time": start_time,
            "end_time": end_time
        }
        
        # 记录任务完成时间，供后续依赖于该任务的任务使用
        completed_tasks[step_id] = end_time

        worker_allocation[step_id] = int(step_id) % max_workers + 1
    
    # 总执行时间为所有任务中结束时间最晚的
    total_time = max(timeline["end_time"] for timeline in task_timelines.values()) if task_timelines else planner_time
    
    return {
        "total_time": total_time,
        "task_timelines": task_timelines,
        "worker_allocation": worker_allocation,
        "task_planning_times": task_available_time  # 返回每个任务的规划完成时间
    }, planner_time

def generate_gantt_chart(task_timelines, max_workers, width=80):
    """
    生成简单的ASCII甘特图
    
    参数:
        task_timelines: 任务时间线
        max_workers: 最大工作线程数
        width: 图表宽度
        
    返回:
        ASCII甘特图文本
    """
    if not task_timelines:
        logger = get_logger()
        logger.warning("没有任务执行数据可供显示甘特图。")
        return "没有任务执行数据可供显示。"
    
    # 确定时间范围
    start_time = min(data["start_time"] for data in task_timelines.values())
    end_time = max(data["end_time"] for data in task_timelines.values())
    time_range = end_time - start_time
    
    # 每个字符表示的时间
    time_per_char = time_range / (width - 20)
    
    # 生成时间轴
    gantt = "时间轴:\n"
    gantt += "0" + " " * (width - 20) + f"{time_range:.2f}s\n"
    gantt += "+" + "-" * (width - 20) + "+\n"
    
    # 生成任务条
    sorted_tasks = sorted(task_timelines.items(), key=lambda x: x[1]["start_time"])
    
    for step_id, data in sorted_tasks:
        rel_start = data["start_time"] - start_time
        rel_end = data["end_time"] - start_time
        
        start_pos = int(rel_start / time_per_char)
        end_pos = int(rel_end / time_per_char)
        
        # 确保每个任务至少显示一个字符
        if start_pos == end_pos:
            end_pos = start_pos + 1
            
        # 限制在图表范围内
        end_pos = min(end_pos, width - 20)
        
        # 构建任务条
        task_bar = " " * start_pos + "#" * (end_pos - start_pos)
        
        # 确保任务条不超过图表宽度
        task_bar = task_bar.ljust(width - 20)
        
        # 添加任务信息
        gantt += f"步骤 {step_id} |{task_bar}| {data['start_time']:.2f}s - {data['end_time']:.2f}s\n"
    
    return gantt




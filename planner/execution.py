import requests
import json
import os
import re
import sys
import time
import pathlib
from collections import defaultdict
import concurrent.futures
from openai import OpenAI
import transformers

from config import ModelConfig, load_config, parse_args
from performance import PerformanceTracker, calculate_performance_metrics

from token_patch import count_tokens_for_model
from log_config import get_logger, log_separator

from router import pre_evaluate_step

def get_api_key(file_path):
    """从文件中获取API密钥"""
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return f.read().strip()
    else:
        logger = get_logger()
        logger.error(f"API密钥文件 '{file_path}' 未找到")
        raise FileNotFoundError(f"API密钥文件 '{file_path}' 未找到")

small_model_client = None
original_split_method = str.split

large_model_client = None
router_model_client = None

# 全局变量
# 用于存储解析后的任务数据
tasks = defaultdict(dict)
current_step = None
xml_buffer = ""

# Track completed steps
completed_steps = set()
# Store futures for each step
futures = {}
# 映射future对象到step_id，用于回调
future_to_id = {}

def initialize_clients(model_config):
    """预先初始化模型客户端"""
    global small_model_client, large_model_client, router_model_client
    if small_model_client is None:
        small_model_client = model_config.get_client(client_type="small")
    if large_model_client is None:
        large_model_client = model_config.get_client(client_type="large")
    if router_model_client is None:
        router_model_client = model_config.get_client(client_type="router")
    print("所有模型客户端已初始化")


def parse_step_attributes(attr_str):
    """解析属性字符串为字典"""
    attrs = {}
    # 使用正则匹配属性键值对
    pattern = r'(\w+)="(.*?)"'
    for match in re.finditer(pattern, attr_str):
        key, value = match.groups()
        attrs[key] = value
    return attrs

def process_xml_buffer():
    """处理XML缓冲区中的完整标签"""
    global xml_buffer, tasks, current_step
    logger = get_logger()
    yaml_config = load_config("config.yaml")
    models_config = yaml_config.get("models", {})
    sequential_execution = models_config.get("sequential_execution", False)
    single_rely = models_config.get("single_rely", False)
    random_router = models_config.get("random_router", False)
    threshold = models_config.get("threshold", 5)
    # 查找完整的<Step>标签
    step_match = re.search(r'<Step\s+(.*?)/>', xml_buffer, re.DOTALL)
    if not step_match:
        # 检查是否有结束标签
        if '</Plan>' in xml_buffer:
            xml_buffer = ""
        return False
    
    full_tag = step_match.group(0)
    attr_str = step_match.group(1).strip()
    
    # 从缓冲区移除已处理的部分
    xml_buffer = xml_buffer[step_match.end():]
    
    # 解析属性
    attrs = parse_step_attributes(attr_str)
    if 'ID' not in attrs:
        return True
    
    # 添加到任务字典
    step_id = attrs['ID']

    # 如果是顺序执行，则强制Rely为所有前面的步骤
    if sequential_execution and not single_rely:
        attrs['Rely'] = ','.join(str(i) for i in range(1, int(attrs['ID'])))
    elif sequential_execution and single_rely:
        if attrs['ID'] == '1':
            attrs['Rely'] = ''
        else:
            attrs['Rely'] = str(int(attrs['ID']) - 1)
    
    if attrs['ID'] == '1':
        attrs['Rely'] = ''
    if 'Difficulty' not in attrs:
        attrs['Difficulty'] = '5'
    
    if not sequential_execution:
        # 处理Rely字段，确保没有依赖未来的步骤，防止死锁
        # 如果没有并行则提供最多的信息保证输出更准确
        if attrs['Rely'] != '':
            Rely = [i.strip() for i in attrs['Rely'].split(',')]
            for i in Rely:
                if int(i) >= int(attrs['ID']):
                    Rely = [j for j in range(1, int(attrs['ID']))]
                    break
            attrs['Rely'] = ','.join(str(i) for i in Rely)

    enable_threshold = models_config.get("enable_threshold", False)
    if not enable_threshold:
        attrs['Difficulty'] = '5'

    if random_router:
        import random
        upper = min(9, threshold + 1)
        lower = max(1, threshold - 1)
        attrs['Difficulty'] = str(random.choice([upper, lower]))

    tasks[step_id] = attrs
    tasks[step_id]['Result'] = None  # 添加Result字段
    logger = get_logger()

    logger.info("================= 解析出新的步骤 =================")
    logger.info(f"步骤 {step_id}: {attrs}")
    logger.info("=================================================")
    print(f"步骤 {step_id}: {attrs}")
    # 设置当前步骤（用于后续结果收集）
    current_step = step_id
    return True

def generate_step_result(prompt, difficulty, model_config, stats_tracker=None, system_prompt=None):
    """生成步骤结果
    
    参数:
        prompt: 提示词
        difficulty: 任务难度
        model_config: 模型配置对象
        stats_tracker: 性能统计跟踪器
    """
    global small_model_client, large_model_client
    yaml_config = load_config("config.yaml")
    models_config = yaml_config.get("models", {})
    executor_config = models_config.get("executor", {})
    temperature = executor_config.get("temperature", 0.0)

    # 根据难度选择模型
    model = model_config.select_model_by_difficulty(difficulty)
    
    # 根据模型选择对应的客户端
    if model == model_config.small_model:
        client = small_model_client
    else:
        client = large_model_client
    
    # 调用API
    try:
        start_time = time.time()
        first_token_time = None
        
        # 使用流式API来测量首个令牌响应时间
        try:
            if system_prompt == None:
                response_stream = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    stream=True
                )
            else:
                response_stream = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    stream=True
                )
            logger = get_logger()
            logger.info("================= 使用流式API生成步骤结果 =================")
            logger.info(f"温度: {temperature}")
            logger.info(f"{model}模型API调用成功，开始接收流式响应")
        except Exception as e:
            # 如果流式API调用失败，尝试使用非流式API
            print(f"流式API调用失败，尝试使用非流式API: {e}")
            logger = get_logger()
            logger.error(f"{model}模型流式API调用失败，尝试使用非流式API: {e}，无法测量TTFT")
            if system_prompt == None:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    stream=False
                )
                logger.info(f"{model}模型API调用成功，开始接收响应")
                logger.info(f"温度: {temperature}")
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    stream=False
                )
                logger.info(f"{model}模型API调用成功，开始接收响应")
                logger.info(f"温度: {temperature}")
            used_time = time.time() - start_time
            # 在这种情况下我们无法测量TTFT
            ttft = None
            logger.info("================= 使用非流式API =================")
            logger.info(f"{model}模型API调用成功，使用时间: {used_time:.2f}秒")
            logger.info(f"模型输出：\n{response.choices[0].message.content}")
            return response.choices[0].message.content
        
        # 收集完整响应
        collected_content = ""
        completion_tokens = 0
        prompt_tokens = 0
        
        for chunk in response_stream:
            if first_token_time is None:
                first_token_time = time.time()
                
            # 从每个块中提取内容并累加
            if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                content = chunk.choices[0].delta.content
                if content:
                    collected_content += content
                    completion_tokens += 1  #
            
        # 计算首个令牌响应时间
        ttft = first_token_time - start_time if first_token_time else None
        
        # 创建一个模拟的完整响应对象
        class MockResponse:
            def __init__(self, content, model_name):
                self.choices = [type('obj', (object,), {
                    'message': type('obj', (object,), {
                        'content': content
                    })
                })]
                
                estimated_prompt_tokens = count_tokens_for_model(prompt, model)
                estimated_completion_tokens = count_tokens_for_model(content, model)
                
                # 确保token数量至少为1
                if estimated_prompt_tokens < 1:
                    estimated_prompt_tokens = 1
                if estimated_completion_tokens < 1:
                    estimated_completion_tokens = 1
                
                self.usage = type('obj', (object,), {
                    'prompt_tokens': estimated_prompt_tokens,
                    'completion_tokens': estimated_completion_tokens,
                    'total_tokens': estimated_prompt_tokens + estimated_completion_tokens
                })
                self.model = model_name
                
        # 使用收集的内容创建模拟响应
        response = MockResponse(collected_content, model)
        used_time = time.time() - start_time
        # 如果没有收集到内容，可能是API调用有问题
        if not collected_content:
            print("警告: 流式API未返回任何内容")
            logger = get_logger()
            logger.error(f"{model}模型流式API未返回任何内容")
        model_name = model_config.small_model if model == model_config.small_model else model_config.large_model
        model_type = "small_model" if model == model_config.small_model else "large_model"
        
        print(f"{model_name} API调用成功，使用时间: {used_time:.2f}秒")
        if ttft is not None:
            print(f"首个令牌响应时间 (TTFT): {ttft:.3f}秒")
            
        # 如果有统计跟踪器，更新token使用统计和TTFT统计
        if stats_tracker and hasattr(response, 'usage'):
            stats_tracker.update_token_usage(
                model_type,
                response.usage.prompt_tokens,
                response.usage.completion_tokens
            )
            # 更新首个令牌响应时间
            if ttft is not None:
                stats_tracker.update_ttft(model_type, ttft)
        
        try:
            return response.choices[0].message.content
        except AttributeError:
            print("错误: 无法从响应中提取内容")
            if hasattr(response, 'choices') and response.choices and hasattr(response.choices[0], 'message'):
                return str(response.choices[0].message)
            return "API调用失败，未能获取有效响应"
    except Exception as e:
        print(f"API调用失败: {e}")
        return f"错误: API调用失败 - {str(e)}"

def build_step_prompt(current_step, tasks, query):
    """构建当前步骤的提示"""
    system_prompt = """
    There is a multiple-choice problem. I need you to solve it and give an answer.
Here is the problem:\n{Problem}

I have broken this problem down into a series of smaller problems. I will assign you sub-problems one by one, and provide the results of the previous sub-problems as a reference for your reasoning.
Please solve the problem and respond according to logic.
"""

    prompt_template = """
    The sub-problem to solve now is: {Task}
    Based on the information above, please provide a concise and clear answer
    {Relied_Results}
    """
    # 如果需要限制token，可以在这里添加
    # prompt_template = """
    # The sub-problem to solve now is: {Task}
    # Based on the information above, please provide a concise and clear answer
    # {Relied_Results}
    # Let's think step by step and use less than {Token} tokens:
    # """

    # 获得依赖的任务的具体结果
    rely_ids = tasks[current_step].get('Rely', '')
    if rely_ids != '':
        relied_results = "\nSo far, the answers to the resolved sub-problems are as follows:"
        relied_results += "\n**CONTEXT (Results from prior steps):**\n"
        # 遍历每个依赖的步骤ID
        for step_id in rely_ids.split(','):
            if step_id in tasks and 'Result' in tasks[step_id] and tasks[step_id]['Result']:
                relied_results += f"Task {step_id}: {tasks[step_id].get('Task', '')} ; \nResult: {tasks[step_id]['Result']} \n"
        relied_results += "\nThey are directly related to this sub-problem, so please pay special attention to them."
    else:
        relied_results = ""
    
    # 如果需要限制token，可以在这里添加，传入token参数
    # return prompt_template.format(
    #     Problem=query,
    #     Task=tasks[current_step].get('Task', ''),
    #     Relied_Results=relied_results,
    #     Token=tasks[current_step].get('Token', '')
    # ), tasks[current_step].get('Difficulty', ''), system_prompt.format(
    #     Problem=query
    # )

    return prompt_template.format(
        Problem=query,
        Task=tasks[current_step].get('Task', ''),
        Relied_Results=relied_results
    ), tasks[current_step].get('Difficulty', ''), system_prompt.format(
        Problem=query
    )

def is_step_ready(step_id, tasks):
    """Check if a step is ready to be processed (dependencies completed or none)"""
    rely_str = tasks[step_id].get('Rely', '')
    if not rely_str:
        return True
    
    rely_steps = rely_str.split(',')
    return all(step in completed_steps for step in rely_steps)

def process_step(step_id, tasks, query, model_config, stats_tracker=None):
    """
    处理单个步骤，并包含健壮的重试机制。
    """
    logger = get_logger()
    
    # 从配置中获取重试参数
    max_attempts = model_config.max_retry_attempts if model_config.enable_retries else 1
    retry_delay = model_config.retry_delay if model_config.enable_retries else 0
    
    # 开始重试循环
    for attempt in range(1, max_attempts + 1):
        try:
            if attempt > 1:
                logger.warning(f"开始重试步骤 {step_id} (第 {attempt}/{max_attempts} 次)...")
                print(f"\n开始重试步骤 {step_id} (第 {attempt}/{max_attempts} 次)...")
                time.sleep(retry_delay)

            print(f"\n开始执行步骤 {step_id}: {tasks[step_id].get('Task', '未知任务')}")
            prompt, difficulty, system_prompt = build_step_prompt(step_id, tasks, query)

            router_config = getattr(model_config, 'router_config', {})
            if router_config.get("enabled", False) and pre_evaluate_step is not None:
                logger.info(f"步骤 {step_id}: 启用智能路由...")
                
                # 2.1. 获取累积指标
                cumulative_tokens = stats_tracker.get_total_tokens() if stats_tracker else 0
                # 注意：这是总运行时间，不是累积模型延迟，但与 hybridflow_runner 逻辑一致
                cumulative_latency = (time.time() - stats_tracker.start_time) if stats_tracker else 0.0
                
                # 2.2. 准备输入
                subproblem = tasks[step_id]
                all_steps_list = sorted(tasks.values(), key=lambda x: int(x.get('ID', 0))) # 确保按ID排序
                
                # 2.3. 调用路由决策
                try:
                    use_cloud, router_metrics = pre_evaluate_step(
                        query, 
                        subproblem, 
                        cumulative_tokens, 
                        cumulative_latency, 
                        all_steps_list, 
                        router_config, 
                        logger
                    )
                except Exception as e:
                    logger.error(f"[Router] 步骤 {step_id} 路由失败: {e}. 默认使用大模型。", exc_info=True)
                    use_cloud = True
                    router_metrics = {"latency_s": 0.0, "error": str(e), "model_type": "router_internal"}

                # 2.4. 记录路由本身的性能
                if stats_tracker and router_metrics:
                    stats_tracker.update_router_latency(router_metrics.get('latency_s', 0))
                    # 记录路由内部的 token 消耗 (例如嵌入)
                    stats_tracker.update_token_usage(
                        "router_internal",
                        router_metrics.get("n_input_tokens", 0),
                        router_metrics.get("n_output_tokens", 0)
                    )
                        
                # 2.5. 根据路由决策覆盖难度
                if use_cloud:
                    # 强制使用大模型
                    difficulty = str(model_config.threshold) # 任何 >= threshold 的值
                    logger.info(f"步骤 {step_id}: 智能路由决策 -> 大模型")
                else:
                    # 强制使用小模型
                    difficulty = "1" # 任何 < threshold 的值
                    logger.info(f"步骤 {step_id}: 智能路由决策 -> 小模型")
            
            elif router_config.get("enabled", False) and pre_evaluate_step is None:
                logger.warning(f"步骤 {step_id}: 智能路由已启用，但 router_logic.py 导入失败。使用Planner建议的难度。")


            model_type = "大模型" if int(difficulty) >= model_config.threshold else "小模型"
            logger.info(f"===== Prompt 给执行器 ({model_type} - 步骤 {step_id}) =====")
            logger.info(system_prompt)
            logger.info(prompt)
            log_separator()

            result = generate_step_result(prompt, difficulty, model_config, stats_tracker, system_prompt)
            
            if not result or (isinstance(result, str) and result.strip().startswith("错误:")):
                raise ValueError(f"步骤 {step_id} 生成了无效或错误的结果: {result}")

            logger.info(f"===== 来自执行器 ({model_type} - 步骤 {step_id}) 的输出 =====")
            logger.info(result)
            log_separator()

            tasks[step_id]['Result'] = result
            completed_steps.add(step_id)
            print(f"步骤 {step_id} 执行成功")
            return step_id  # 成功执行，立即退出循环并返回

        except Exception as e:
            error_message = f"步骤 {step_id} 执行出错 (第 {attempt}/{max_attempts} 次): {e}"
            logger.error(error_message)
            print(error_message)
            
            if attempt == max_attempts:
                tasks[step_id]['Result'] = f"错误: 经过 {max_attempts} 次尝试后依然失败 - {e}"
                completed_steps.add(step_id)
                logger.critical(f"步骤 {step_id} 所有重试均失败，已标记为错误状态以避免阻塞。")
                return step_id

    return step_id

def completion_callback(future):
    """处理完成任务的回调函数
    
    参数:
        future: 已完成的Future对象
    """
    global future_to_id, futures, completed_steps, tasks
    
    # 获取对应的step_id
    step_id = future_to_id.get(future)
    if not step_id:
        print("警告: 无法找到与Future对应的任务ID")
        return
        
    try:
        # 获取结果，这里不会阻塞因为任务已完成
        future.result()
        print(f"回调中: 步骤 {step_id} 已完成")
        # 标记为已完成
        completed_steps.add(step_id)
    except Exception as e:
        print(f"回调中: 任务 {step_id} 执行错误: {e}")
        # 即使出错也标记为完成，避免死锁
        completed_steps.add(step_id)
        tasks[step_id]['Result'] = f"错误: {str(e)}"
    finally:
        # 从跟踪字典中移除
        if future in future_to_id:
            del future_to_id[future]
        if step_id in futures:
            del futures[step_id]

def router(tasks, model_config, query, executor, stats_tracker=None):
    """调度任务并行执行"""
    global futures, future_to_id
    has_new_tasks = False
    
    # 计算可以执行的任务及其优先级
    ready_tasks = []
    for step_id in tasks:
        if step_id not in futures and step_id not in completed_steps:
            if is_step_ready(step_id, tasks):
                # 优先级计算：基于依赖任务的数量和任务ID（较早的任务优先）
                rely_str = tasks[step_id].get('Rely', '')
                rely_count = len(rely_str.split(',')) if rely_str else 0
                priority = (rely_count, int(step_id))
                ready_tasks.append((step_id, priority))
    
    # 按优先级排序（依赖少的先执行）
    ready_tasks.sort(key=lambda x: x[1])
    
    # 批量提交任务到执行器
    for step_id, _ in ready_tasks:
        print(f"调度步骤 {step_id}: {tasks[step_id].get('Task', '未知任务')}")
        future = executor.submit(process_step, step_id, tasks, query, model_config, stats_tracker)
        # 添加回调
        future.add_done_callback(completion_callback)
        # 保存映射关系
        futures[step_id] = future
        future_to_id[future] = step_id
        has_new_tasks = True
    
    completed_future_ids = [step_id for step_id, f in list(futures.items()) if f.done()]
    if completed_future_ids:
        has_new_tasks = True

    if futures:
        return True
    
    return has_new_tasks

def print_results(tasks):
    """打印任务执行结果
    
    参数:
        tasks: 任务字典
    """
    print("\n\n执行结果:")
    for step_id, attrs in sorted(tasks.items(), key=lambda x: int(x[0])):
        print(f"\n--- 步骤 {step_id}: {attrs.get('Task', '未知任务')} ---")
        if 'Result' in attrs and attrs['Result']:
            print(attrs['Result'])
        else:
            print("(无结果)")
        print(f"--- 步骤 {step_id} 结束 ---")
    
def wait_for_completion_and_get_final_result(tasks, query, config, stats_tracker=None):
    """等待所有任务完成并返回最终结果
    
    参数:
        tasks: 任务字典
        query: 原始查询
        config: 模型配置
        stats_tracker: 性能统计跟踪器
        
    返回:
        最终结果字符串
    """
    logger = get_logger()

    # 确保所有任务已完成
    if not all(task.get('Result') is not None for task in tasks.values()):
        print("等待所有任务完成...")
        # time.sleep(1)
    
    # 按步骤ID排序
    sorted_tasks = sorted(tasks.items(), key=lambda x: int(x[0]))
    
    # 构建最终结果
    final_result = "# 问题求解最终结果\n\n"
    
    final_result += f"## 原始问题\n{query}\n\n"
    
    # 添加解决方案步骤
    final_result += "## 解决步骤\n\n"
    for step_id, attrs in sorted_tasks:
        final_result += f"### 步骤 {step_id}: {attrs.get('Task', '未知任务')}\n"
        if 'Result' in attrs and attrs['Result']:
            final_result += f"{attrs['Result']}\n\n"
        else:
            final_result += "（此步骤未完成）\n\n"
    
    # 已经完成了所有的subtask,向router小模型询问最终结果
    prompt = """Based on the results of all steps below, please provide only the final answer. No other explanations or details are needed.

    PROBLEM:
    {query}

    SOLUTION STEPS:
    {steps}
    """
    
    # 构建步骤结果文本
    steps_text = ""
    for step_id, attrs in sorted_tasks:
        step_text = f"步骤 {step_id}: {attrs.get('Task', '未知任务')}\n"
        if 'Result' in attrs and attrs['Result']:
            step_text += f"{attrs['Result']}\n\n"
        else:
            step_text += "（此步骤未完成）\n\n"
        steps_text += step_text
    
    # 调用小模型获取最终答案
    try:
        final_prompt = prompt.format(query=query, steps=steps_text)

        logger.info("===== Prompt 给最终总结模型 (小模型) =====")
        logger.info(final_prompt)
        log_separator()

        final_answer = generate_step_result(final_prompt, "1", config, stats_tracker)  # 使用小模型（难度为1，低于阈值）

        logger.info("===== 来自最终总结模型 (小模型) 的输出 =====")
        logger.info(final_answer)
        log_separator()

        final_result += f"## 最终答案\n{final_answer}\n"
        # 停止性能跟踪
        stats_tracker.stop_tracking()

    except Exception as e:
        print(f"获取最终答案时出错: {e}")
        # 如果失败，回退到使用最后一个步骤的结果
        if sorted_tasks:
            last_step_id, last_step = sorted_tasks[-1]
            final_result += f"## 最终答案\n"
            if 'Result' in last_step and last_step['Result']:
                final_result += f"{last_step['Result']}\n"
                # 停止性能跟踪
                stats_tracker.stop_tracking()
            else:
                final_result += "（未能获得最终答案）\n"
                # 停止性能跟踪
                stats_tracker.stop_tracking()
    
    return final_result

def generate_task_dependency_report(tasks):
    """生成任务依赖关系报告
    
    参数:
        tasks: 任务字典
    
    返回:
        任务依赖关系报告文本
    """
    report = "# 任务规划依赖关系\n\n"
    
    # 按步骤ID排序
    sorted_tasks = sorted(tasks.items(), key=lambda x: int(x[0]))
    
    report += "| 步骤ID | 任务描述 | 依赖步骤 | 难度 | Token限制 |\n"
    report += "| ------ | -------- | -------- | ---- | --------- |\n"
    
    for step_id, attrs in sorted_tasks:
        task_desc = attrs.get('Task', '未知任务')
        rely = attrs.get('Rely', '无')
        difficulty = attrs.get('Difficulty', '未指定')
        token_limit = attrs.get('Token', '未指定')
        
        report += f"| {step_id} | {task_desc} | {rely} | {difficulty} | {token_limit} |\n"
    
    return report

def dataset_run_parallel_execution(query, solution, config, workers=4, dataset_build_config=None):
    """
    为构建数据集生成规划。
    给定一个问题和参考答案，利用规划模型生成训练数据。

    参数:
        query: 要解决的问题
        solution: 用于指导planner的参考答案
        config: 模型配置对象
        workers: 并行工作线程数 (此函数中未使用)
        dataset_build_config: 包含数据集构建设置的字典
    
    返回:
        一个元组 (full_plan_with_thinking, plan_only, system_prompt)。
        失败时返回 (None, None, None)。
    """
    global xml_buffer, tasks, router_model_client
    
    logger = get_logger()

    # 如果未提供，则使用默认构建配置
    if dataset_build_config is None:
        dataset_build_config = {
            'use_ground_truth_to_guide_planner': True,
            'save_thinking': True,
        }

    # 确保客户端已初始化
    initialize_clients(config)
    
    # 为此运行重置状态
    xml_buffer = ""
    tasks = defaultdict(dict)
    
    print(f"正在为问题构建数据集: {query[:100]}...")
    logger.info(f"正在为问题构建数据集: {query[:100]}...")

    # 根据是本地还是远程路由模型定义系统提示
    if config.use_local_router:
        system_prompt = """You are an expert AI cognitive scientist and systems architect. Your mission is to analyze a given problem and create a structured XML plan that follows the **Explain-Analyze-Generate (EAG)** framework.

The plan will be executed by a multi-threaded AI system using two distinct models. Your task decomposition and difficulty assignments must leverage the unique capabilities of these models.

### **Executor Model Profiles**
You will be assigning tasks to two available models. Use their profiles below to accurately estimate the `Difficulty` for each step:
  * **Small Model (Llama 3.2 3B): A highly capable and efficient small model. It excels at tasks requiring strong instruction-following, summarization, and rewriting. It is proficient in standard grade-school math (GSM8K) and common-sense reasoning (ARC Challenge). Use this model for well-defined, procedural, or structured tasks.
  * **Large Model (GPT-4o): A powerful, state-of-the-art large model with a vast knowledge base. It demonstrates superior performance in tasks requiring deep reasoning and expert-level knowledge, such as advanced scientific reasoning (GPQA Diamond, MMLU-Pro), competition-level math (AIME), and complex coding (LiveCodeBench). It is the preferred choice for tasks requiring synthesis, critical analysis, and solving problems in specialized domains.

### **Plan Structure: The EAG Framework**
Your generated `<Plan>` **must** be structured into three logical stages:

1.  **Step 1: The "Explain" Step**
      * The plan must begin with a single, foundational step (ID="1"). These steps should not have any rely (i.e. Rely="").
      * **Task:** The task for this step is directly inspired by the Explainer agent's role. It must be phrased as follows:
        > **"To assist the following agents, what is your understanding of the question after reviewing it, focusing only on essential information and filtering out all irrelevant details?"**

2.  **The "Analyze" Steps**
      * These are the intermediate steps that perform the core logical work.
      * **Task:** Break down the problem into the smallest possible, independent sub-tasks to solve the problem. These steps should rely on the "Explain" step or other completed analysis steps.
      * **Core Directives for Plan Generation**:
        1.  **Analyze the Problem**: Break down the problem into its core logical components.
        2.  **Strategic Milestones**: Focus on the high-level, conceptual milestones required to solve the problem.
        3.  **Maximize Parallelism**: Decompose into as many independent sub-tasks as possible.
        4.  **Formulate Actionable Questions**: The `Task` attribute must be a clear, self-contained question ending with a question mark (?). Do not leak answers in the task description.
        5.  **Delegate Knowledge Retrieval**: If a specific formula, theorem, or principle is required, your task is to create a step that **asks for that formula or principle** (e.g., "What is the formula for...?"). Delegate the retrieval of specific knowledge to the Executor.
      * **Goal:** Maximize parallelism. If multiple pieces of information can be processed independently, create a separate step for each.

3.  **The Final "Generate" Step**
      * The plan must conclude with a single aggregation step.
      * **Task:** The task for this step is directly inspired by the Generator agent's role. It must be phrased as follows:
        > **"After reviewing the original question and the thoughts of previous agents, what is the final answer to the question?"**
**Keep the plan Concise**: The final plan must contain **fewer than 7 steps**. Focus only on the most critical milestones needed to solve the problem.

### **XML Plan Constraints**
  * `ID`: A unique integer.
  * `Task`: The question for the executor AI. Must end with a question mark (?).
  * `Difficulty`: An integer from 1-9.
      * **1-4 (Small Model):** Procedural tasks, basic calculations, applying a known formula.
      * **5-9 (Large Model):** Complex reasoning, synthesis, or critical knowledge retrieval.
  * `Token`: An estimated integer for the answer's token count.
  * `Rely`: The `ID`(s) of prerequisite steps, separated by commas if multiple.

### Examples
**Problem**: Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.\\n\\nA. 0\\nB. 4\\nC. 2\\nD. 6
**Output**:
<Plan>
<Step ID="1" Task="To assist the following agents, what is your understanding of the question after reviewing it, focusing only on essential information and filtering out all irrelevant details?" Difficulty="4" Token="50" Rely=""/>
<Step ID="2" Task="Is there a dependency between sqrt(2), sqrt(3), and sqrt(18)? Simplify the field extension Q(sqrt(2), sqrt(3), sqrt(18)) if possible." Difficulty="3" Token="30" Rely="1"/>
<Step ID="3" Task="Based on the simplified field extension from Step 2, what is the degree of this extension over Q?" Difficulty="5" Token="30" Rely="2"/>
<Step ID="4" Task="After reviewing the original question and the thoughts of previous agents, what is the final answer to the question?" Difficulty="2" Token="20" Rely="3"/>
</Plan>

**Problem**: The set of all real numbers under the usual multiplication operation is not a group since\\n\\nA. multiplication is not a binary operation\\nB. multiplication is not associative\\nC. identity element does not exist\\nD. zero has no inverse
**Output**:
<Plan>
<Step ID="1" Task="To assist the following agents, what is your understanding of the question after reviewing it, focusing only on essential information and filtering out all irrelevant details?" Difficulty="3" Token="50" Rely=""/>
<Step ID="2" Task="Check the closure property: Is multiplication a binary operation on the set of all real numbers?" Difficulty="2" Token="20" Rely="1"/>
<Step ID="3" Task="Check the associative property: Is multiplication of real numbers associative?" Difficulty="2" Token="20" Rely="1"/>
<Step ID="4" Task="Check the identity property: Is there an identity element for multiplication in the set of real numbers?" Difficulty="2" Token="20" Rely="1"/>
<Step ID="5" Task="Check the inverse property: Does every element in the set of real numbers have a multiplicative inverse?" Difficulty="3" Token="30" Rely="1"/>
<Step ID="6" Task="After reviewing the original question and the thoughts of previous agents, what is the final answer to the question?" Difficulty="4" Token="30" Rely="2,3,4,5"/>
</Plan>
Now, based on the following Problem, generate a response that meets all the requirements above. The final plan must contain **fewer than 7 steps**. 
"""
    else:
        system_prompt = """You are an expert AI cognitive scientist and systems architect. Your mission is to analyze a given problem and create a structured XML plan that follows the **Explain-Analyze-Generate (EAG)** framework.

The plan will be executed by a multi-threaded AI system using two distinct models. Your task decomposition and difficulty assignments must leverage the unique capabilities of these models.

### **Executor Model Profiles**
You will be assigning tasks to two available models. Use their profiles below to accurately estimate the `Difficulty` for each step:
  * **Small Model (Llama 3.2 3B): A highly capable and efficient small model. It excels at tasks requiring strong instruction-following, summarization, and rewriting. It is proficient in standard grade-school math (GSM8K) and common-sense reasoning (ARC Challenge). Use this model for well-defined, procedural, or structured tasks.
  * **Large Model (GPT-4o): A powerful, state-of-the-art large model with a vast knowledge base. It demonstrates superior performance in tasks requiring deep reasoning and expert-level knowledge, such as advanced scientific reasoning (GPQA Diamond, MMLU-Pro), competition-level math (AIME), and complex coding (LiveCodeBench). It is the preferred choice for tasks requiring synthesis, critical analysis, and solving problems in specialized domains.

### **Plan Structure: The EAG Framework**
Your generated `<Plan>` **must** be structured into three logical stages:

1.  **Step 1: The "Explain" Step**
      * The plan must begin with a single, foundational step (ID="1").
      * **Task:** The task for this step is directly inspired by the Explainer agent's role. It must be phrased as follows:
        > **"To assist the following agents, what is your understanding of the question after reviewing it, focusing only on essential information and filtering out all irrelevant details?"**

2.  **The "Analyze" Steps**
      * These are the intermediate steps that perform the core logical work.
      * **Task:** Break down the problem into the smallest possible, independent sub-tasks to solve the problem. These steps should rely on the "Explain" step (i.e., `Rely="1"`) or other completed analysis steps.
      * **Core Directives for Plan Generation**:
        1.  **Analyze the Problem**: Break down the problem into its core logical components.
        2.  **Strategic Milestones**: Focus on the high-level, conceptual milestones required to solve the problem.
        3.  **Maximize Parallelism**: Decompose into as many independent sub-tasks as possible.
        4.  **Formulate Actionable Questions**: The `Task` attribute must be a clear, self-contained question ending with a question mark (?). Do not leak answers in the task description.
        5.  **Delegate Knowledge Retrieval**: If a specific formula, theorem, or principle is required, your task is to create a step that **asks for that formula or principle** (e.g., "What is the formula for...?"). Delegate the retrieval of specific knowledge to the Executor.
      * **Goal:** Maximize parallelism. If multiple pieces of information can be processed independently, create a separate step for each.

3.  **The Final "Generate" Step**
      * The plan must conclude with a single aggregation step.
      * **Task:** The task for this step is directly inspired by the Generator agent's role. It must be phrased as follows:
        > **"After reviewing the original question and the thoughts of previous agents, what is the final answer to the question?"**
**Keep the plan Concise**: The final plan must contain **fewer than 7 steps**. Focus only on the most critical milestones needed to solve the problem.

### **XML Plan Constraints**
  * `ID`: A unique integer.
  * `Task`: The question for the executor AI. Must end with a question mark (?).
  * `Difficulty`: An integer from 1-9.
      * **1-4 (Small Model):** Procedural tasks, basic calculations, applying a known formula.
      * **5-9 (Large Model):** Complex reasoning, synthesis, or critical knowledge retrieval.
  * `Token`: An estimated integer for the answer's token count.
  * `Rely`: The `ID`(s) of prerequisite steps, separated by commas if multiple.

### Examples
**Problem**: Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.\\n\\nA. 0\\nB. 4\\nC. 2\\nD. 6
**Output**:
<Plan>
<Step ID="1" Task="To assist the following agents, what is your understanding of the question after reviewing it, focusing only on essential information and filtering out all irrelevant details?" Difficulty="4" Token="50" Rely=""/>
<Step ID="2" Task="Is there a dependency between sqrt(2), sqrt(3), and sqrt(18)? Simplify the field extension Q(sqrt(2), sqrt(3), sqrt(18)) if possible." Difficulty="3" Token="30" Rely="1"/>
<Step ID="3" Task="Based on the simplified field extension from Step 2, what is the degree of this extension over Q?" Difficulty="5" Token="30" Rely="2"/>
<Step ID="4" Task="After reviewing the original question and the thoughts of previous agents, what is the final answer to the question?" Difficulty="2" Token="20" Rely="3"/>
</Plan>

**Problem**: The set of all real numbers under the usual multiplication operation is not a group since\\n\\nA. multiplication is not a binary operation\\nB. multiplication is not associative\\nC. identity element does not exist\\nD. zero has no inverse
**Output**:
<Plan>
<Step ID="1" Task="To assist the following agents, what is your understanding of the question after reviewing it, focusing only on essential information and filtering out all irrelevant details?" Difficulty="3" Token="50" Rely=""/>
<Step ID="2" Task="Check the closure property: Is multiplication a binary operation on the set of all real numbers?" Difficulty="2" Token="20" Rely="1"/>
<Step ID="3" Task="Check the associative property: Is multiplication of real numbers associative?" Difficulty="2" Token="20" Rely="1"/>
<Step ID="4" Task="Check the identity property: Is there an identity element for multiplication in the set of real numbers?" Difficulty="2" Token="20" Rely="1"/>
<Step ID="5" Task="Check the inverse property: Does every element in the set of real numbers have a multiplicative inverse?" Difficulty="3" Token="30" Rely="1"/>
<Step ID="6" Task="After reviewing the original question and the thoughts of previous agents, what is the final answer to the question?" Difficulty="4" Token="30" Rely="2,3,4,5"/>
</Plan>
Now, based on the following Problem, generate a response that meets all the requirements above. The final plan must contain **fewer than 7 steps**. 
"""   
    # 根据是否使用真实答案构建用户查询
    if dataset_build_config.get('use_ground_truth_to_guide_planner', True):
        user_query = f'''
            Question: {query}
            Solution: {solution}
            Please generate a solution plan for the question in XML format you can use the solution as a reference.
        '''
    else:
        user_query = f'''
            Question: {query}
            Plan:
        '''

    logger.info("===== Prompt to Planner (Router) for Dataset Building =====")
    logger.info(f"System Prompt:\n{system_prompt}")
    logger.info(f"User Query:\n{user_query}")
    log_separator()

    full_completion = ""
    try:
        # 选择客户端和模型
        client = router_model_client
        model = config.local_router_model if config.use_local_router else config.router_model
        
        # API 调用
        response_stream = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            stream=True,
            temperature=1
        )

        for chunk in response_stream:
            if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                content = chunk.choices[0].delta.content
                if content:
                    full_completion += content
    
    except Exception as e:
        logger.error(f"Error during planner API call for dataset building: {e}", exc_info=True)
        print(f"\nError during planner API call: {e}")
        return None, None, None

    logger.info("===== Full Output from Planner (Router) =====")
    logger.info(full_completion)
    log_separator()

    # 根据 `save_thinking` 配置处理输出
    plan_only = ""
    plan_match = re.search(r'<Plan>.*</Plan>', full_completion, re.DOTALL)
    if plan_match:
        plan_only = plan_match.group(0)

    # 如果输出中没有 <think> 标签，则带思考的完整输出就是只有 plan
    if "<think>" not in full_completion:
        full_completion_with_think = plan_only
    else:
        full_completion_with_think = full_completion

    return full_completion_with_think, plan_only, system_prompt


def run_parallel_execution(query, config, workers=4, process=None):
    """运行并行执行流程
    
    参数:
        query: 要解决的问题
        config: 模型配置对象
        workers: 并行工作线程数
        
    # 重置收集的输出内容
    reset_collected_output()
    返回:
        (tasks, stats_tracker, full_completion): 任务字典, 性能跟踪器, 和规划器的完整输出
    """
    global xml_buffer, tasks, completed_steps, futures, router_model_client

    logger = get_logger()
    
    # 创建性能统计跟踪器
    stats_tracker = PerformanceTracker(config)
    
    # 初始化所有客户端
    initialize_clients(config)
    
    # 重置全局状态
    xml_buffer = ""
    tasks = defaultdict(dict)
    completed_steps = set()
    futures = {}
    future_to_id = {}
    
    # 创建线程池
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
    
    # 初始化变量跟踪解析进度
    task_count = 0
    print("开始处理问题：", query)
    print("正在获取解决方案计划...")

    if config.use_local_router:
        system_prompt = """You are an expert AI cognitive scientist and systems architect. Your mission is to analyze a given problem and create a structured XML plan that follows the **Explain-Analyze-Generate (EAG)** framework.

The plan will be executed by a multi-threaded AI system using two distinct models. Your task decomposition and difficulty assignments must leverage the unique capabilities of these models.

### **Executor Model Profiles**
You will be assigning tasks to two available models. Use their profiles below to accurately estimate the `Difficulty` for each step:
  * **Small Model (Llama 3.2 3B): A highly capable and efficient small model. It excels at tasks requiring strong instruction-following, summarization, and rewriting. It is proficient in standard grade-school math (GSM8K) and common-sense reasoning (ARC Challenge). Use this model for well-defined, procedural, or structured tasks.
  * **Large Model (GPT-4o): A powerful, state-of-the-art large model with a vast knowledge base. It demonstrates superior performance in tasks requiring deep reasoning and expert-level knowledge, such as advanced scientific reasoning (GPQA Diamond, MMLU-Pro), competition-level math (AIME), and complex coding (LiveCodeBench). It is the preferred choice for tasks requiring synthesis, critical analysis, and solving problems in specialized domains.

### **Plan Structure: The EAG Framework**
Your generated `<Plan>` **must** be structured into three logical stages:

1.  **Step 1: The "Explain" Step**
      * The plan must begin with a single, foundational step (ID="1").
      * **Task:** The task for this step is directly inspired by the Explainer agent's role. It must be phrased as follows:
        > **"To assist the following agents, what is your understanding of the question after reviewing it, focusing only on essential information and filtering out all irrelevant details?"**

2.  **The "Analyze" Steps**
      * These are the intermediate steps that perform the core logical work.
      * **Task:** Break down the problem into the smallest possible, independent sub-tasks to solve the problem. These steps should rely on the "Explain" step (i.e., `Rely="1"`) or other completed analysis steps.
      * **Core Directives for Plan Generation**:
        1.  **Analyze the Problem**: Break down the problem into its core logical components.
        2.  **Strategic Milestones**: Focus on the high-level, conceptual milestones required to solve the problem.
        3.  **Maximize Parallelism**: Decompose into as many independent sub-tasks as possible.
        4.  **Formulate Actionable Questions**: The `Task` attribute must be a clear, self-contained question ending with a question mark (?). Do not leak answers in the task description.
        5.  **Delegate Knowledge Retrieval**: If a specific formula, theorem, or principle is required, your task is to create a step that **asks for that formula or principle** (e.g., "What is the formula for...?"). Delegate the retrieval of specific knowledge to the Executor.
      * **Goal:** Maximize parallelism. If multiple pieces of information can be processed independently, create a separate step for each.

3.  **The Final "Generate" Step**
      * The plan must conclude with a single aggregation step.
      * **Task:** The task for this step is directly inspired by the Generator agent's role. It must be phrased as follows:
        > **"After reviewing the original question and the thoughts of previous agents, what is the final answer to the question?"**
**Keep the plan Concise**: The final plan must contain **fewer than 7 steps**. Focus only on the most critical milestones needed to solve the problem.

### **XML Plan Constraints**
  * `ID`: A unique integer.
  * `Task`: The question for the executor AI. Must end with a question mark (?).
  * `Difficulty`: An integer from 1-9.
      * **1-4 (Small Model):** Procedural tasks, basic calculations, applying a known formula.
      * **5-9 (Large Model):** Complex reasoning, synthesis, or critical knowledge retrieval.
  * `Token`: An estimated integer for the answer's token count.
  * `Rely`: The `ID`(s) of prerequisite steps, separated by commas if multiple.

### Examples
**Problem**: Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.\\n\\nA. 0\\nB. 4\\nC. 2\\nD. 6
**Output**:
<Plan>
<Step ID="1" Task="To assist the following agents, what is your understanding of the question after reviewing it, focusing only on essential information and filtering out all irrelevant details?" Difficulty="4" Token="50" Rely=""/>
<Step ID="2" Task="Is there a dependency between sqrt(2), sqrt(3), and sqrt(18)? Simplify the field extension Q(sqrt(2), sqrt(3), sqrt(18)) if possible." Difficulty="3" Token="30" Rely="1"/>
<Step ID="3" Task="Based on the simplified field extension from Step 2, what is the degree of this extension over Q?" Difficulty="5" Token="30" Rely="2"/>
<Step ID="4" Task="After reviewing the original question and the thoughts of previous agents, what is the final answer to the question?" Difficulty="2" Token="20" Rely="3"/>
</Plan>

**Problem**: The set of all real numbers under the usual multiplication operation is not a group since\\n\\nA. multiplication is not a binary operation\\nB. multiplication is not associative\\nC. identity element does not exist\\nD. zero has no inverse
**Output**:
<Plan>
<Step ID="1" Task="To assist the following agents, what is your understanding of the question after reviewing it, focusing only on essential information and filtering out all irrelevant details?" Difficulty="3" Token="50" Rely=""/>
<Step ID="2" Task="Check the closure property: Is multiplication a binary operation on the set of all real numbers?" Difficulty="2" Token="20" Rely="1"/>
<Step ID="3" Task="Check the associative property: Is multiplication of real numbers associative?" Difficulty="2" Token="20" Rely="1"/>
<Step ID="4" Task="Check the identity property: Is there an identity element for multiplication in the set of real numbers?" Difficulty="2" Token="20" Rely="1"/>
<Step ID="5" Task="Check the inverse property: Does every element in the set of real numbers have a multiplicative inverse?" Difficulty="3" Token="30" Rely="1"/>
<Step ID="6" Task="After reviewing the original question and the thoughts of previous agents, what is the final answer to the question?" Difficulty="4" Token="30" Rely="2,3,4,5"/>
</Plan>
Now, based on the following Problem, generate a response that meets all the requirements above. The final plan must contain **fewer than 7 steps**. 
"""
        user_prompt = f'''**Problem**: {query}
**Output**:
        '''
    else:
        system_prompt = """You are an expert AI cognitive scientist and systems architect. Your mission is to analyze a given problem and create a structured XML plan that follows the **Explain-Analyze-Generate (EAG)** framework.

The plan will be executed by a multi-threaded AI system using two distinct models. Your task decomposition and difficulty assignments must leverage the unique capabilities of these models.

### **Executor Model Profiles**
You will be assigning tasks to two available models. Use their profiles below to accurately estimate the `Difficulty` for each step:
  * **Small Model (Llama 3.2 3B): A highly capable and efficient small model. It excels at tasks requiring strong instruction-following, summarization, and rewriting. It is proficient in standard grade-school math (GSM8K) and common-sense reasoning (ARC Challenge). Use this model for well-defined, procedural, or structured tasks.
  * **Large Model (GPT-4o): A powerful, state-of-the-art large model with a vast knowledge base. It demonstrates superior performance in tasks requiring deep reasoning and expert-level knowledge, such as advanced scientific reasoning (GPQA Diamond, MMLU-Pro), competition-level math (AIME), and complex coding (LiveCodeBench). It is the preferred choice for tasks requiring synthesis, critical analysis, and solving problems in specialized domains.

### **Plan Structure: The EAG Framework**
Your generated `<Plan>` **must** be structured into three logical stages:

1.  **Step 1: The "Explain" Step**
      * The plan must begin with a single, foundational step (ID="1").
      * **Task:** The task for this step is directly inspired by the Explainer agent's role. It must be phrased as follows:
        > **"To assist the following agents, what is your understanding of the question after reviewing it, focusing only on essential information and filtering out all irrelevant details?"**

2.  **The "Analyze" Steps**
      * These are the intermediate steps that perform the core logical work.
      * **Task:** Break down the problem into the smallest possible, independent sub-tasks to solve the problem. These steps should rely on the "Explain" step (i.e., `Rely="1"`) or other completed analysis steps.
      * **Core Directives for Plan Generation**:
        1.  **Analyze the Problem**: Break down the problem into its core logical components.
        2.  **Strategic Milestones**: Focus on the high-level, conceptual milestones required to solve the problem.
        3.  **Maximize Parallelism**: Decompose into as many independent sub-tasks as possible.
        4.  **Formulate Actionable Questions**: The `Task` attribute must be a clear, self-contained question ending with a question mark (?). Do not leak answers in the task description.
        5.  **Delegate Knowledge Retrieval**: If a specific formula, theorem, or principle is required, your task is to create a step that **asks for that formula or principle** (e.g., "What is the formula for...?"). Delegate the retrieval of specific knowledge to the Executor.
      * **Goal:** Maximize parallelism. If multiple pieces of information can be processed independently, create a separate step for each.

3.  **The Final "Generate" Step**
      * The plan must conclude with a single aggregation step.
      * **Task:** The task for this step is directly inspired by the Generator agent's role. It must be phrased as follows:
        > **"After reviewing the original question and the thoughts of previous agents, what is the final answer to the question?"**
**Keep the plan Concise**: The final plan must contain **fewer than 7 steps**. Focus only on the most critical milestones needed to solve the problem.

### **XML Plan Constraints**
  * `ID`: A unique integer.
  * `Task`: The question for the executor AI. Must end with a question mark (?).
  * `Difficulty`: An integer from 1-9.
      * **1-4 (Small Model):** Procedural tasks, basic calculations, applying a known formula.
      * **5-9 (Large Model):** Complex reasoning, synthesis, or critical knowledge retrieval.
  * `Token`: An estimated integer for the answer's token count.
  * `Rely`: The `ID`(s) of prerequisite steps, separated by commas if multiple.

### Examples
**Problem**: Four years ago, Kody was only half as old as Mohamed. If Mohamed is currently twice 30 years old, how old is Kody?
**Output**:
<Plan>
<Step ID="1" Task="To assist the following agents, what is your understanding of the question after reviewing it, focusing only on essential information and filtering out all irrelevant details?" Difficulty="3" Token="60" Rely=""/>
<Step ID="2" Task="Based on the explanation in Step 1, what is Mohamed's current age?" Difficulty="2" Token="10" Rely="1"/>
<Step ID="3" Task="Using Mohamed's current age from Step 2, what was his age four years ago?" Difficulty="1" Token="10" Rely="2"/>
<Step ID="4" Task="Based on Mohamed's age four years ago, what was Kody's age four years ago?" Difficulty="2" Token="10" Rely="3"/>
<Step ID="5" Task="After reviewing the original question and the thoughts of previous agents, what is the final answer to the question?" Difficulty="2" Token="15" Rely="4"/>
</Plan>

**Question**: "Which of the following stars or stellar systems will appear the brightest in V magnitude when observed from Earth? Assume there is no extinction. [List of 6 options with apparent/absolute magnitudes and distances]"
**Output**:
<Plan>
<Step ID="1" Task="What is the formula for calculating the combined apparent magnitude of a multi-star system from the individual apparent magnitudes of its components?" Difficulty="2" Rely=""/>
<Step ID="2" Task="What is the distance modulus formula that relates a star's apparent magnitude (m), its absolute magnitude (M), and its distance in parsecs (d)?" Difficulty="2" Rely=""/>
<Step ID="3" Task="For each of the six options (a-f) provided in the problem, calculate its final apparent V magnitude as observed from Earth. Use the formulas from Step 1 and 2 where necessary. List the final apparent magnitude for each option." Difficulty="6" Rely="1,2"/>
<Step ID="4" Task="Based on the six apparent magnitude values calculated in Step 3, which star or stellar system is the brightest (i.e., has the numerically lowest magnitude value)?" Difficulty="2" Rely="3"/>
</Plan>

**Question**: "Identify the compound C9H11NO2 using the given data. IR: medium to strong intensity bands at 3420 cm-1, 3325 cm-1; strong band at 1720 cm-1. 1H NMR: 1.20 ppm (t, 3H); 4.0 ppm (bs, 2H); 4.5 ppm (q, 2H); 7.0 ppm (d, 2H), 8.0 ppm (d, 2H)."
**Output**:
<Plan>
<Step ID="1" Task="What functional group(s) are indicated by the IR absorption bands at 3420, 3325, and 1720 cm-1?" Difficulty="4" Rely=""/>
<Step ID="2" Task="In the 1H NMR spectrum, what structural fragment is suggested by the combination of a triplet signal at 1.20 ppm (3H) and a quartet signal at 4.5 ppm (2H)?" Difficulty="4" Rely=""/>
<Step ID="3" Task="In the 1H NMR spectrum, what structural feature is suggested by the presence of two distinct doublet signals at 7.0 ppm (2H) and 8.0 ppm (2H) in the aromatic region?" Difficulty="5" Rely=""/>
<Step ID="4" Task="What does the broad singlet signal at 4.0 ppm (2H) in the 1H NMR spectrum, combined with the IR data from Step 1, suggest about the functional group present?" Difficulty="5" Rely="1"/>
<Step ID="5" Task="Based on the fragments identified in the previous steps (a para-substituted aromatic ring, an ethyl group, and an amine/ester/amide functional group), assemble a complete molecular structure that matches the formula C9H11NO2." Difficulty="7" Rely="1,2,3,4"/>
<Step ID="6" Task="Compare the structure deduced in Step 5 with the provided options to identify the correct compound name." Difficulty="2" Rely="5"/>
</Plan>

**Problem**: Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.\\n\\nA. 0\\nB. 4\\nC. 2\\nD. 6
**Output**:
<Plan>
<Step ID="1" Task="To assist the following agents, what is your understanding of the question after reviewing it, focusing only on essential information and filtering out all irrelevant details?" Difficulty="4" Token="50" Rely=""/>
<Step ID="2" Task="Is there a dependency between sqrt(2), sqrt(3), and sqrt(18)? Simplify the field extension Q(sqrt(2), sqrt(3), sqrt(18)) if possible." Difficulty="3" Token="30" Rely="1"/>
<Step ID="3" Task="Based on the simplified field extension from Step 2, what is the degree of this extension over Q?" Difficulty="5" Token="30" Rely="2"/>
<Step ID="4" Task="After reviewing the original question and the thoughts of previous agents, what is the final answer to the question?" Difficulty="2" Token="20" Rely="3"/>
</Plan>

**Problem**: The set of all real numbers under the usual multiplication operation is not a group since\\n\\nA. multiplication is not a binary operation\\nB. multiplication is not associative\\nC. identity element does not exist\\nD. zero has no inverse
**Output**:
<Plan>
<Step ID="1" Task="To assist the following agents, what is your understanding of the question after reviewing it, focusing only on essential information and filtering out all irrelevant details?" Difficulty="3" Token="50" Rely=""/>
<Step ID="2" Task="Check the closure property: Is multiplication a binary operation on the set of all real numbers?" Difficulty="2" Token="20" Rely="1"/>
<Step ID="3" Task="Check the associative property: Is multiplication of real numbers associative?" Difficulty="2" Token="20" Rely="1"/>
<Step ID="4" Task="Check the identity property: Is there an identity element for multiplication in the set of real numbers?" Difficulty="2" Token="20" Rely="1"/>
<Step ID="5" Task="Check the inverse property: Does every element in the set of real numbers have a multiplicative inverse?" Difficulty="3" Token="30" Rely="1"/>
<Step ID="6" Task="After reviewing the original question and the thoughts of previous agents, what is the final answer to the question?" Difficulty="4" Token="30" Rely="2,3,4,5"/>
</Plan>

Now, based on the following Problem, generate a response that meets all the requirements above. The final plan must contain **fewer than 7 steps**. 
"""
        user_prompt = f'''**Problem**: {query}
**Output**:
        '''
    logger.info("===== Prompt 给 Planner (Router) =====")
    logger.info(f"System Prompt:\n{system_prompt}")
    logger.info(f"User Prompt:\n{user_prompt}")
    log_separator()
    
    full_completion = "" 
    # 使用预初始化的路由模型客户端
    yaml_config = load_config("config.yaml")
    models_config = yaml_config.get("models", {})
    planner_model_client = models_config.get("planner", {})

    temperature = planner_model_client.get("temperature", 0.0)
    top_p = planner_model_client.get("top_p", 0.9)
    max_tokens = planner_model_client.get("max_tokens", 500)
    enable_thinking = planner_model_client.get("enable_thinking", False)

    try:
        # 记录路由模型开始生成计划的时间，用于计算首个令牌响应时间
        router_start_time = time.time()
        first_token_received = False
        ttft = None
        
        # 根据配置决定是使用本地路由模型还是远程路由模型
        if config.use_local_router:
            response_stream = router_model_client.chat.completions.create(
                model=config.local_router_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                stream=True,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                extra_body={"enable_thinking": enable_thinking}
            )
            logger.info(f"----------------- 使用本地路由模型: {config.local_router_model} -----------------")
            logger.info(f"温度: {temperature}, top_p: {top_p}, max_tokens: {max_tokens}, enable_thinking: {enable_thinking}")
        else:
            response_stream = router_model_client.chat.completions.create(
                model=config.router_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                stream=True,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                extra_body={"enable_thinking": enable_thinking}
            )
            logger.info(f"----------------- 使用远程路由模型: {config.router_model} -----------------")
            logger.info(f"温度: {temperature}, top_p: {top_p}, max_tokens: {max_tokens}, enable_thinking: {enable_thinking}")
        model = config.local_router_model if config.use_local_router else config.router_model
        # 使用deepseek_v3_tokenizer计算tokens
        system_prompt_tokens = count_tokens_for_model(system_prompt, model)
        user_prompt_tokens = count_tokens_for_model(user_prompt, model)
        prompt_tokens = system_prompt_tokens + user_prompt_tokens

        logger.info(f"----- Router (Planner) Prompt Token Usage -----")
        logger.info(f"Prompt Tokens: {prompt_tokens}")
        logger.info(f"System Prompt Tokens: {system_prompt_tokens}")
        logger.info(f"User Prompt Tokens: {user_prompt_tokens}")

        completion_tokens = 0
        
        for chunk in response_stream:
            if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                content = chunk.choices[0].delta.content
                if content:
                    # 记录首个token响应时间
                    if not first_token_received:
                        ttft = time.time() - router_start_time
                        first_token_received = True
                        if stats_tracker:
                            stats_tracker.update_ttft("router_model", ttft)
                    
                    print(content, end="", flush=True)  # 实时输出
                    full_completion += content # 累加内容
                    
                    # 使用deepseek_v3_tokenizer更新完成tokens计数
                    completion_tokens += count_tokens_for_model(content, model)
                    
                    # 添加到XML缓冲区
                    xml_buffer += content
                    
                    # 尝试解析缓冲区中的完整标签
                    parsed_count = 0
                    while process_xml_buffer():
                        parsed_count += 1
                        task_count += 1
                    
                    # 只有在解析到新任务时才启动路由
                    if parsed_count > 0:
                        print(f"\n已解析 {task_count} 个任务，启动任务调度...")
                        router(tasks, config, query, executor, stats_tracker)
        
        logger.info("===== 来自 Planner (Router) 的输出 =====")
        logger.info(full_completion)
        logger.info(f"router模型完成的Token数: {completion_tokens}")
        
        log_separator()

        # 更新router模型的token使用情况
        if stats_tracker:
            stats_tracker.update_token_usage("router_model", prompt_tokens, completion_tokens)
            stats_tracker.save_planner_output(prompt_tokens, completion_tokens, ttft)
                        
    except Exception as e:
        print(f"\n处理响应时出错: {e}")
    
    print(f"\n计划生成完成，共解析 {task_count} 个任务")
    
    # 继续处理可能的剩余XML标签
    while process_xml_buffer():
        pass
    
    # 处理所有剩余任务直到全部完成
    print("\n\n开始执行所有任务...")
    while tasks and any(step_id not in completed_steps for step_id in tasks):
        if not router(tasks, config, query, executor, stats_tracker):
            if futures:
                concurrent.futures.wait(list(futures.values()))
            else:
                break

    # 关闭线程池
    executor.shutdown(wait=True)
    
    return tasks, stats_tracker, full_completion

def judge_correct(question, gold_answer, final_answer, model_config):
    """根据真实结果判断最终答案是否正确
    
    参数:
        question: 问题文本
        gold_answer: 金标准答案
        final_answer: 最终生成的答案
        model_config: 模型配置对象
        
    返回:
        是否正确的布尔值和判断结果文本
    """
    # model_name = model_config.small_model
    # client = model_config.get_client(client_type="small")

    model_name = "deepseek-chat"
    client = OpenAI(api_key=get_api_key('ApiKeys/deepseek'), base_url="https://api.deepseek.com")

    print(f"--------------------------调用{model_name}根据真实答案判断答案正确性--------------------------")
    # prompt = f"""
    # Your task is to strictly compare the 'Student's Answer' to the 'Standard Answer'.

    # **IMPORTANT: Your decision must be based exclusively on whether the final answers are equivalent. Completely ignore the problem statement and any reasoning.**

    # - **Problem**: {question}
    # - **Standard Answer**: {gold_answer}
    # - **Student's Answer**: {final_answer}

    # If the 'Student's Answer' matches the 'Standard Answer', output only the word `True`. Otherwise, output only the word `False`. Do not provide any explanation.
    # """
    prompt = f"""Here is a math problem with a standard answer and a student's solution. Please help me determine if the student's solution is correct. If the numerical value are same, then it is correct.
                               
                Problem: {question}

                Standard answer: {gold_answer}

                Answer: {final_answer}

                If the student's answer is correct, just output True; otherwise, just output False.
                No explanation is required.
    """
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            stream=False
        )
        
        result_text = response.choices[0].message.content.strip()
        # 解析结果文本，确定是否正确
        is_correct = "true" in result_text.lower() and "false" not in result_text.lower()
        logger = get_logger()
        logger.info("===== 来自 judge_correct 的输出（有真实结果）=====")
        logger.info(f"问题: {question}\n标准答案: {gold_answer}\n答案: {final_answer}\n判断结果: {result_text}")
        return is_correct, result_text
    except Exception as e:
        logger = get_logger()
        logger.error(f"判断答案正确性时出错: {e}\n问题: {question}\n标准答案: {gold_answer}\n答案: {final_answer}")
        return False, f"判断错误: {str(e)}"

def LLM_judge(question, final_answer, model_config):
    """使用大模型对没有真实答案的问题判断是否正确
    
    参数:
        question: 问题文本
        final_answer: 最终生成的答案
        model_config: 模型配置对象
        
    返回:
        是否正确的布尔值和判断结果文本
    """
    model_name = model_config.large_model
    client = model_config.get_client(client_type="large")

    # model_name = "deepseek-chat"
    # client = OpenAI(api_key=get_api_key('usage/deepseek'), base_url="https://api.deepseek.com")

    print(f"--------------------------调用{model_name}判断答案正确性（没有真实答案）--------------------------")
    prompt = f"""Here is a math problem and a student's solution. Please help me determine if the student's solution is correct. If the numerical value are same, then it is correct.
                               
                Problem: {question}

                Answer: {final_answer}

                If the student's answer is correct, just output True; otherwise, just output False.
                No explanation is required.
    """
    
    # 使用全局预初始化的客户端，而不是每次创建新客户端
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            stream=False
        )
        
        result_text = response.choices[0].message.content.strip()
        # 解析结果文本，确定是否正确
        is_correct = "true" in result_text.lower() and "false" not in result_text.lower()
        logger = get_logger()
        logger.info("===== 来自 LLM_judge 的输出（没有真实结果）=====")
        logger.info(f"问题: {question}\n答案: {final_answer}\n判断结果: {result_text}")
        return is_correct, result_text
    except Exception as e:
        logger.info("===== 来自 LLM_judge 的输出（没有真实结果）=====")
        logger.error(f"{model_name}判断答案正确性时出错: {e}\n问题: {question}\n答案: {final_answer}，判断错误: {str(e)}")
        return False, f"判断错误: {str(e)}"
    
def judge_question_difficulty(question, model_config):
    """判断问题难度
    
    参数:
        question: 问题文本
        model_config: 模型配置对象
        
    返回:
        问题难度（字符串）
    """
    if model_config.enable_threshold:
        prompt = f"""You are an expert AI system architect. Your task is to analyze a given problem and assign a difficulty score from 1 to 10. This score will determine whether the problem is routed to a smaller, faster AI model or a larger, more powerful one.
    Refer to the model profiles below to make your decision.
    Executor Model Profiles:
    Small Model (Difficulty 1-4): This model is highly efficient for tasks that are procedural or rely on well-defined knowledge.
        * Strengths: Basic calculations, direct information retrieval (e.g., "What is the capital of France?"), definition lookups, and following simple, explicit instructions.
    Large Model (Difficulty 5-10): This model has a vast knowledge base and excels at complex reasoning, analysis, and synthesis.
        * Strengths: Multi-step reasoning, in-depth analysis, comparing and contrasting complex ideas, solving nuanced problems, and synthesizing information from multiple sources to draw a conclusion.
    Based on the problem below, determine which model is better suited to solve it and assign a corresponding difficulty score.
    Problem: {question}
    Instructions:
    Output only the integer representing the difficulty level. Do not provide any other text or explanation.
        """

        client = model_config.get_client(client_type="large")
        try:
            response = client.chat.completions.create(
                model=model_config.large_model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                stream=False
            )
            
            difficulty = response.choices[0].message.content.strip()
            print(f"问题难度判断结果: {difficulty}")
            return difficulty
        except Exception as e:
            logger = get_logger()
            logger.warning(f"判断问题难度时出错: {e}\n问题: {question}，按照高于阈值处理")
            return str(model_config.threshold + 1)
    else:
        print("----------未启用难度阈值，默认使用模型协作模式----------")
        logger = get_logger()
        logger.info("----------未启用难度阈值，默认使用模型协作模式----------")
        return str(model_config.threshold + 1)

def call_small_model_directly(question, model_config, stats_tracker=None):
    """直接调用小模型进行处理
    
    参数:
        question: 问题文本
        model_config: 模型配置对象
        stats_tracker: 性能统计跟踪器（可选）
        
    返回:
        小模型的响应内容
    """
    logger = get_logger()
    # 构建提示词
    prompt = """You are a problem-solving assistant. I will provide you with a problem. Your task is to solve it step by step and provide the final answer.
PROBLEM:
{question}
""".format(question=question)
    logger.info("===== Prompt 给执行器 (小模型 - 直接调用) =====")
    logger.info(prompt)
    log_separator()
    result = generate_step_result(prompt, "1", model_config, stats_tracker)
    logger.info("===== 来自执行器 (小模型 - 直接调用) 的输出 =====")
    logger.info(result)
    log_separator()
    return result

def call_large_model_directly(question, model_config, stats_tracker=None):
    """直接调用大模型进行处理
    
    参数:
        question: 问题文本
        model_config: 模型配置对象
        stats_tracker: 性能统计跟踪器（可选）
        
    返回:
        大模型的响应内容
    """
    # 构建提示词
    prompt = """You are a problem-solving assistant. I will provide you with a problem. Your task is to solve it step by step and provide the final answer.
PROBLEM:
{question}
""".format(question=question)
    return generate_step_result(prompt, "10", model_config, stats_tracker)

def build_report_path(base_dir="data_reports", is_dataset=False, dataset_name="", config=None, timestamp=None):
    """构建层次化的报告路径
    
    参数:
        base_dir: 基础目录，默认为data_reports
        is_dataset: 是否为数据集报告(True)或单个问题报告(False)
        dataset_name: 数据集名称，仅当is_dataset=True时有效
        config: 模型配置对象
        timestamp: 时间戳，如果为None则自动生成
        
    返回:
        完整的目录路径
    """
    if timestamp is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
    # 获取模型名称，避免路径中的非法字符
    def clean_name(name):
        if name is None:
            return "unknown"
        # 提取模型名称的核心部分
        if "/" in name:
            name = name.split("/")[-1]
        # 移除可能导致路径问题的字符
        return ''.join(c for c in name if c.isalnum() or c in '_-.')
    
    # 获取模型名称
    router_name = "local_router" if config and config.use_local_router else clean_name(config.router_model if config else None)
    large_model = clean_name(config.large_model if config else None)
    small_model = clean_name(config.small_model if config else None)
    
    # 构建路径
    path_parts = [base_dir]
    
    if is_dataset:
        # 数据集路径结构: data_reports/dataset/数据集名称/router/large/small/时间戳
        dataset_name = os.path.basename(dataset_name) if dataset_name else "unknown_dataset"
        path_parts.extend(["dataset", dataset_name, router_name, large_model, small_model, timestamp])
    else:
        # 单个问题路径结构: data_reports/single/router/large/small/时间戳
        path_parts.extend(["single", router_name, large_model, small_model, timestamp])
    
    # 构建完整路径
    full_path = os.path.join(*path_parts)
    
    # 确保目录存在
    os.makedirs(full_path, exist_ok=True)
    
    return full_path

def save_result_to_file(final_result, config, workers, correctness_report, performance_report, dependency_report, theoretical_report=None):
    """将结果保存到文件
    
    参数:
        final_result: 最终结果文本
        config: 模型配置对象
        workers: 工作线程数
        correctness_report: 正确性报告
        performance_report: 性能报告
        dependency_report: 依赖关系报告
        theoretical_report: 理论性能报告（可选）
    """
    try:
        # 使用新的层次化目录结构
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = build_report_path(
            base_dir="data_reports", 
            is_dataset=False, 
            config=config, 
            timestamp=timestamp
        )
        
        # 使用简化的文件名
        output_file = os.path.join(output_dir, "result.md")
        
        model_usage = f"使用小模型: {config.small_model}\n\n使用大模型: {config.large_model}\n\n使用路由模型: {config.router_model}\n\n"
        threshold_info = f"难度阈值: {config.threshold}\n\n工作线程数: {workers}\n\n"
        model_usage += threshold_info
        
        # 将性能报告、依赖关系报告和理论性能报告添加到最终结果中
        final_result_with_stats = model_usage + "\n\n" + final_result + "\n\n" + correctness_report + performance_report + "\n\n" + dependency_report
        
        # 如果有理论性能报告，也添加进去
        if theoretical_report:
            final_result_with_stats += "\n\n" + theoretical_report

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(final_result_with_stats)
        print(f"结果已保存至: {output_file}")
        return output_file
    except Exception as e:
        print(f"保存结果时出错: {e}")
        return None
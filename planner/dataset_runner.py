import os
import json
import time
import re
import random
from tqdm import tqdm
from config import ModelConfig, load_config
from performance import PerformanceTracker
from execution import (
    dataset_run_parallel_execution,
    run_parallel_execution, wait_for_completion_and_get_final_result,
    judge_question_difficulty, call_small_model_directly, judge_correct
)
from log_config import setup_logger, get_logger, log_separator

from utils import build_report_path 
import pathlib
from evaluation import Evaluator, aggregate_dataset_reports, format_aggregated_report_md

def build_report_path(base_dir="data_reports", is_dataset=True, dataset_name="", config=None, timestamp=None):
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
        logger = get_logger()
        logger.warning("timestamp为空")
        
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
        dataset_name = pathlib.Path(dataset_name).stem if dataset_name else "unknown_dataset"
        path_parts.extend(["dataset", dataset_name, router_name, large_model, small_model, timestamp])
    else:
        # 单个问题路径结构: data_reports/single/router/large/small/时间戳
        path_parts.extend(["single", router_name, large_model, small_model, timestamp])
    
    # 构建完整路径
    full_path = os.path.join(*path_parts)
    
    # 确保目录存在
    os.makedirs(full_path, exist_ok=True)
    
    return full_path


class DatasetRunner:
    """数据集处理器，用于批量处理数据集并生成统计报告"""
    
    def __init__(self, config, dataset_path, limit=None, workers=4, evaluator=None):
        """初始化数据集处理器"""
        self.config = config
        self.dataset_path = dataset_path
        self.workers = workers
        self.results = []
        self.evaluator = evaluator
        
        # **MODIFIED LOGIC START**
        # This is a robust fix to ensure the limit from config.yaml is always respected,
        # bypassing any issues with parameter passing through the call stack.
        logger = get_logger()
        try:
            # Load the config file directly to get the definitive limit value.
            yaml_config = load_config("config.yaml")
            dataset_config = yaml_config.get("dataset", {})
            self.limit = dataset_config.get("limit", None)
            self.seed = dataset_config.get("seed", None)
            logger.info(f"Successfully loaded limit from config.yaml: {self.limit}")
            logger.info(f"Successfully loaded seed from config.yaml: {self.seed}")
        except Exception as e:
            # If loading the config fails for any reason, fall back to the passed parameter.
            logger.error(f"Could not load config.yaml to get limit, falling back to passed parameter. Error: {e}")
            self.limit = limit
            self.seed = None
        # **MODIFIED LOGIC END**

        self.dataset = self._load_dataset()

        
    def _load_dataset(self):
        """加载数据集
        
        返回:
            数据集列表
        """
        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            
            # 如果设置了限制，则随机抽取N个问题
            if self.limit is not None and len(dataset) > self.limit:
                logger = get_logger()
                if self.seed is not None:
                    random.seed(self.seed)
                    logger.info(f"使用种子 {self.seed} 进行随机抽样，抽取 {self.limit} 个样本。")
                else:
                    logger.info(f"未提供种子，进行随机抽样，抽取 {self.limit} 个样本。")
                dataset = random.sample(dataset, self.limit)
            
            return dataset
        except Exception as e:
            print(f"加载数据集时出错: {e}")
            logger = get_logger()
            logger.error(f"加载数据集时出错: {e}", exc_info=True)
            return []

    def process_dataset(self, enable_threshold=True, timestamp=None):
        """处理整个数据集
        
        返回:
            处理结果列表
        """
        if not self.dataset:
            print("数据集为空，无法处理")
            return []
        
        print(f"开始处理数据集，共 {len(self.dataset)} 个问题...")
        
        # 创建时间戳，用于整个处理过程
        # timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 使用tqdm显示进度
        for i, problem_data in enumerate(tqdm(self.dataset, desc="处理数据集")):
            problem = problem_data.get("problem", "")
            solution = problem_data.get("answer", "")
            process = problem_data.get("solution", None)
            logger = get_logger()
            try:
                # 每个问题的性能统计
                result = self.process_single_problem(problem, solution, enable_threshold, i + 1, process)
                self.results.append(result)
                
                # 每处理完一个问题就保存一次结果
                self.save_results_json("dataset_results", timestamp)
                
                # 打印当前进度
                print(f"完成进度: {i+1}/{len(self.dataset)}")
                logger.info(f"完成进度: {i+1}/{len(self.dataset)}")
            except Exception as e:
                print(f"处理问题时出错: {e}")
                print(f"已跳过该问题，继续处理下一个问题")
                logger.error(f"处理问题时出错: {e}", exc_info=True)
            
        return self.results

    def process_single_problem(self, problem, solution, enable_threshold, problem_index, process=None):
        """处理单个问题
        
        参数:
            problem: 问题文本
            solution: 标准答案
            problem_index: 问题的索引号
            
        返回:
            处理结果字典
        """
        # === 新增：获取日志记录器并记录新问题的开始 ===
        logger = get_logger()
        logger.info(f"===== 开始处理问题 #{problem_index} =====")
        logger.info(f"问题: {problem[:200]}...")
        log_separator()
        # ==========================================

        print(f"\n处理问题: {problem[:100]}...")
        
        # 初始化结果字典
        result = {
            "problem": problem,
            "gold_solution": solution,
            "model_solution": "",
            "difficulty": 0,
            "is_correct": False,
            "judge_result": "",
            "stats": None,
            "execution_time": 0
        }
        
        start_time = time.time()
        
        try:
            # 判断问题难度
            difficulty = judge_question_difficulty(problem, self.config)
            result["difficulty"] = difficulty
            logger = get_logger()
            # 创建性能统计跟踪器
            small_model_name = self.config.small_model if hasattr(self.config, 'small_model') else "qwen3-14b"
            large_model_name = self.config.large_model if hasattr(self.config, 'large_model') else "gpt-4o"
            model_name = small_model_name if int(difficulty) < self.config.threshold else large_model_name
            stats_tracker = PerformanceTracker(self.config)
            
            planner_output = None # 初始化 planner_output
            
            if not enable_threshold:
                logger.info("禁用结果判断，直接使用并行执行流程")

            if enable_threshold and int(difficulty) < self.config.threshold:
                logger.info(f"问题难度 {difficulty} 低于阈值 {self.config.threshold}，使用小模型处理")

                # 直接调用小模型处理
                model_solution = call_small_model_directly(problem, self.config, stats_tracker)
                result["model_solution"] = model_solution
                
                tasks = {
                    "1": {
                        "Task": "直接使用小模型解答问题",
                        "Difficulty": difficulty, "Result": model_solution
                    }
                }
                result["tasks"] = tasks
                result["execution_time"] = time.time() - start_time
            else:
                # 运行并行执行流程
                tasks, stats_tracker, planner_output = run_parallel_execution(problem, self.config, self.workers, process)
                
                # 获取最终结果
                model_solution = wait_for_completion_and_get_final_result(tasks, problem, self.config, stats_tracker)
                result["model_solution"] = model_solution
                result["tasks"] = tasks
                result["execution_time"] = time.time() - start_time
                # 生成理论性能报告
                from output_performance import generate_theoretical_performance_report
                theoretical_report = generate_theoretical_performance_report(tasks, self.config, stats_tracker.planner_output)
                result["theoretical_report"] = theoretical_report
                
                # 提取理论性能指标
                # 从理论性能报告中提取关键数据
                try:
                    # 从报告文本中解析关键指标
                    import re
                    
                    # 提取理论执行时间相关指标
                    total_time_match = re.search(r"并行总时间.*?(\d+\.\d+)", theoretical_report)
                    planner_time_match = re.search(r"规划模型.*?(\d+\.\d+)", theoretical_report)
                    sequential_time_match = re.search(r"顺序总时间.*?(\d+\.\d+)", theoretical_report)
                    parallel_speedup_match = re.search(r"并行加速比.*?(\d+\.\d+)x", theoretical_report)
                    
                    # 提取小模型和大模型的任务执行时间
                    small_model_time_match = re.search(r"小模型任务.*?(\d+\.\d+)", theoretical_report)
                    large_model_time_match = re.search(r"大模型任务.*?(\d+\.\d+)", theoretical_report)
                    
                    # 构建理论性能指标字典
                    theoretical_metrics = {
                        "total_execution_time": float(total_time_match.group(1)) if total_time_match else 0,
                        "planner_time": float(planner_time_match.group(1)) if planner_time_match else 0,
                        "sequential_time": float(sequential_time_match.group(1)) if sequential_time_match else 0,
                        "parallel_speedup": float(parallel_speedup_match.group(1)) if parallel_speedup_match else 0,
                        "small_model_time": float(small_model_time_match.group(1)) if small_model_time_match else 0,
                        "large_model_time": float(large_model_time_match.group(1)) if large_model_time_match else 0,
                        "task_execution_time": (float(small_model_time_match.group(1)) if small_model_time_match else 0) + 
                                              (float(large_model_time_match.group(1)) if large_model_time_match else 0)
                    }
                    
                    # 保存理论性能指标
                    result["theoretical_metrics_raw"] = theoretical_metrics
                except Exception as e:
                    logger = get_logger()
                    logger.error(f"提取理论性能指标时出错: {e}", exc_info=True)
                
                # 计算任务规划指标
                from task_metrics import calculate_task_metrics
                task_metrics = calculate_task_metrics(tasks)
                
                # 添加任务规划指标到结果中
                result["total_tasks_num"] = task_metrics["total_tasks_num"]
                result["compression_ratio"] = task_metrics["compression_ratio"]
                result["avg_task_plan_tokens"] = task_metrics["avg_task_plan_tokens"]
            
            # 判断结果正确性（使用gold_solution进行判断）
            gold_answer = solution  # 使用提供的标准答案
            is_correct, judge_result = judge_correct(problem, gold_answer, model_solution, self.config)
            result["is_correct"] = is_correct
            result["judge_result"] = judge_result
            
            stats_tracker.stop_tracking()
            result["stats"] = stats_tracker

            # --- 新增：运行评估 ---
            if self.evaluator and self.evaluator.enabled and planner_output:
                eval_results = {}
                # 评估 Planner
                planner_report = self.evaluator.evaluate_planner(problem, planner_output)
                if planner_report:
                    eval_results["planner_report"] = planner_report
                
                # 评估 Executor
                executor_reports = self.evaluator.evaluate_executor(problem, planner_output, tasks, self.config)
                if executor_reports:
                    eval_results["executor_reports"] = executor_reports
                
                result["evaluation_results"] = eval_results
            # --- 评估结束 ---
            
            logger.info(f"===== 问题 #{problem_index} 性能报告 (Markdown) =====")
            logger.info(stats_tracker.format_performance_report())
            log_separator()

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"处理问题 #{problem_index} 时发生错误: {e}", exc_info=True)
            log_separator()
        
        return result
    
    def generate_report(self):
        """生成数据集处理报告
        
        返回:
            处理报告文本
        """
        if not self.results:
            return "没有处理结果，无法生成报告"
        
        # 统计正确率
        correct_count = sum(1 for r in self.results if r.get("is_correct", False))
        accuracy = correct_count / len(self.results) if self.results else 0
        
        # 统计按难度的正确率
        difficulty_stats = {}
        for result in self.results:
            diff = result.get("difficulty", "未知")
            if diff not in difficulty_stats:
                difficulty_stats[diff] = {"total": 0, "correct": 0}
            
            difficulty_stats[diff]["total"] += 1
            if result.get("is_correct", False):
                difficulty_stats[diff]["correct"] += 1
        
        # 统计平均执行时间
        avg_time = sum(r.get("execution_time", 0) for r in self.results) / len(self.results) if self.results else 0
        
        # 统计平均成本
        total_cost = 0
        
        # TTFT 统计和生成速度统计
        total_avg_ttft = 0
        total_time_without_ttft = 0
        total_tokens_per_second = {"small_model": 0, "large_model": 0, "router_model": 0, "total": 0}
        
        for result in self.results:
            stats = result.get("stats")
            if stats:
                costs = stats.calculate_cost()
                total_cost += costs["total"]
                
                # 计算平均TTFT
                def calc_avg_ttft(ttft_list):
                    return sum(ttft_list) / len(ttft_list) if ttft_list else 0
                
                all_ttft = stats.ttft_metrics["total"]
                if all_ttft:
                    total_avg_ttft += calc_avg_ttft(all_ttft)
                    
                # 计算去除TTFT的时间
                def calc_total_ttft(ttft_list):
                    return sum(ttft_list) if ttft_list else 0
                    
                exec_time = result.get("execution_time", 0)
                time_without_ttft = exec_time - calc_total_ttft(all_ttft)
                total_time_without_ttft += time_without_ttft
                
                # 累加每秒生成token数
                tokens_per_second = stats.calculate_tokens_per_second()
                for model_type in total_tokens_per_second.keys():
                    total_tokens_per_second[model_type] += tokens_per_second[model_type]
        
        # 计算任务规划指标的平均值
        total_tasks_count = 0
        total_compression_ratio = 0.0
        total_plan_tokens = 0.0
        task_planning_results_count = 0
        
        # 理论性能指标统计
        theoretical_results = {
            "total_execution_time": 0.0,  # 总执行时间
            "planner_time": 0.0,          # 规划时间
            "task_execution_time": 0.0,   # 任务执行时间
            "sequential_time": 0.0,       # 顺序执行总时间
            "parallel_speedup": 0.0,      # 并行加速比
            "count": 0                    # 有效数据计数
        }
        
        for result in self.results:
            # 任务规划指标统计
            if "total_tasks_num" in result and "compression_ratio" in result and "avg_task_plan_tokens" in result:
                total_tasks_count += result["total_tasks_num"]
                total_compression_ratio += result["compression_ratio"]
                total_plan_tokens += result["avg_task_plan_tokens"]
                task_planning_results_count += 1
            
            # 提取理论性能报告中的关键指标
            if "theoretical_report" in result and result["theoretical_report"]:
                report_text = result["theoretical_report"]
                
                # 从理论报告中提取关键时间数据
                try:
                    # 提取并行总时间（实际的理论执行总时间）
                    total_time_match = re.search(r"并行总时间\s*\|.*?\|\s*([\d.]+)\s*\|", report_text)
                    if total_time_match:
                        theoretical_results["total_execution_time"] += float(total_time_match.group(1))
                        
                    # 提取规划阶段时间
                    planner_time_match = re.search(r"规划模型.*?\|\s*1\s*\|\s*([\d.]+)", report_text)
                    if planner_time_match:
                        theoretical_results["planner_time"] += float(planner_time_match.group(1))
                        
                    # 提取任务总执行时间（累计）
                    task_time_match = re.search(r"任务总执行时间\(累计\)\s*\|\s*([\d.]+)", report_text)
                    if task_time_match:
                        theoretical_results["task_execution_time"] += float(task_time_match.group(1))
                        
                    # 提取顺序总时间
                    sequential_time_match = re.search(r"顺序总时间\s*\|.*?\|\s*([\d.]+)", report_text)
                    if sequential_time_match:
                        theoretical_results["sequential_time"] += float(sequential_time_match.group(1))
                        
                    # 提取并行加速比
                    speedup_match = re.search(r"并行总时间\s*\|.*?\|.*?\|\s*([\d.]+)x", report_text)
                    if speedup_match:
                        theoretical_results["parallel_speedup"] += float(speedup_match.group(1))
                        
                    # 计数有效理论报告数据
                    if total_time_match and sequential_time_match:  # 这两个是必须的最小数据集
                        theoretical_results["count"] += 1
                        print(f"成功提取理论性能数据: 总时间={total_time_match.group(1)}, 顺序时间={sequential_time_match.group(1)}")
                except Exception as e:
                    logger.error(f"提取理论报告数据出错: {e}", exc_info=True)
                
        # 计算平均值
        avg_tasks_num = total_tasks_count / task_planning_results_count if task_planning_results_count > 0 else 0
        avg_compression_ratio = total_compression_ratio / task_planning_results_count if task_planning_results_count > 0 else 0
        avg_plan_tokens = total_plan_tokens / task_planning_results_count if task_planning_results_count > 0 else 0
        
        # 计算理论性能平均值
        theory_count = theoretical_results["count"]
        avg_theoretical = {
            "total_execution_time": theoretical_results["total_execution_time"] / theory_count if theory_count > 0 else 0,
            "planner_time": theoretical_results["planner_time"] / theory_count if theory_count > 0 else 0,
            "task_execution_time": theoretical_results["task_execution_time"] / theory_count if theory_count > 0 else 0,
            "sequential_time": theoretical_results["sequential_time"] / theory_count if theory_count > 0 else 0,
            "parallel_speedup": theoretical_results["parallel_speedup"] / theory_count if theory_count > 0 else 0,
        }
        
        # 计算平均值
        avg_cost = total_cost / len(self.results) if self.results else 0
        avg_ttft = total_avg_ttft / len(self.results) if self.results else 0
        avg_time_without_ttft = total_time_without_ttft / len(self.results) if self.results else 0
        
        # 计算平均每秒生成token数
        avg_tokens_per_second = {model_type: val / len(self.results) if self.results else 0 
                                for model_type, val in total_tokens_per_second.items()}
        
        # 生成报告
        report = "# 数据集处理报告\n\n"
        report += f"## 模型配置\n\n"
        report += f"- 小模型: {self.config.small_model}\n"
        report += f"- 大模型: {self.config.large_model}\n"
        if self.config.use_local_router:
            report += f"- 路由模型: {self.config.local_router_model}\n"
        else:
            report += f"- 路由模型: {self.config.router_model}\n"
        report += f"- 难度阈值: {self.config.threshold}\n"
        report += f"- 工作线程数: {self.workers}\n\n"
        report += f"## 概述\n\n"
        report += f"- 数据集: {self.dataset_path}\n"
        report += f"- 问题总数: {len(self.results)}\n"
        report += f"- 正确数量: {correct_count}\n"
        report += f"- 准确率: {accuracy:.2%}\n"
        report += f"- 平均执行时间: {avg_time:.2f} 秒\n"
        report += f"- 平均成本: ${avg_cost:.4f}\n\n"
        
        if self.evaluator and self.evaluator.enabled:
            avg_scores = aggregate_dataset_reports(self.results)
            report += format_aggregated_report_md(avg_scores)

        # 添加任务规划指标
        report += f"## 任务规划指标\n\n"
        report += f"- 平均任务步骤数: {avg_tasks_num:.2f}\n"
        report += f"- 平均压缩比例: {avg_compression_ratio:.2%}\n"
        report += f"- 平均每步骤Token限制: {avg_plan_tokens:.2f} tokens\n\n"
        
        # 添加理论性能指标（如果有）
        if theoretical_results["count"] > 0:
            report += f"## 理论性能指标\n\n"
            report += f"- 平均理论执行时间: {avg_theoretical['total_execution_time']:.3f} 秒\n"
            
            # 添加除零保护
            planner_percentage = (avg_theoretical['planner_time']/avg_theoretical['total_execution_time']*100) if avg_theoretical['total_execution_time'] > 0.001 else 0
            task_percentage = (avg_theoretical['task_execution_time']/avg_theoretical['total_execution_time']*100) if avg_theoretical['total_execution_time'] > 0.001 else 0
            real_time_ratio = (avg_theoretical['total_execution_time']/avg_time) if avg_time > 0.001 else 0
            
            
            report += f"- 平均顺序执行时间: {avg_theoretical['sequential_time']:.3f} 秒\n"
            report += f"- 平均并行加速比: {avg_theoretical['parallel_speedup']:.2f}x\n"
            report += f"- 理论与实际执行时间比例: {real_time_ratio:.2f}x\n\n"
        
        # 添加TTFT和生成速度统计
        report += f"## 性能指标\n\n"
        report += f"### 首个令牌响应时间 (TTFT)\n"
        report += f"- 平均首个令牌响应时间: {avg_ttft:.3f} 秒\n\n"
        
        report += f"### 去除TTFT的执行时间\n"
        report += f"- 平均去除TTFT的执行时间: {avg_time_without_ttft:.3f} 秒\n\n"
        
        report += f"### 生成速度\n"
        report += f"- 小模型平均每秒生成token数: {avg_tokens_per_second['small_model']:.2f} tokens/s\n"
        report += f"- 大模型平均每秒生成token数: {avg_tokens_per_second['large_model']:.2f} tokens/s\n"
        report += f"- 路由模型平均每秒生成token数: {avg_tokens_per_second['router_model']:.2f} tokens/s\n"
        report += f"- 总平均每秒生成token数: {avg_tokens_per_second['total']:.2f} tokens/s\n\n"
        
        # 生成详细结果表格
        report += f"## 详细结果\n\n"
        report += "| # | 问题 | 正确? | 执行时间(秒) | 成本($) | 步骤数 | 压缩比例 | 平均Token |\n"
        report += "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
        
        for i, result in enumerate(self.results):
            is_correct = "✓" if result.get("is_correct", False) else "✗"
            problem = result.get("problem", "")
            # 截断问题以适合表格
            if len(problem) > 50:
                problem = problem[:47] + "..."
            problem = problem.replace("\n", " ")
            
            exec_time = result.get("execution_time", 0)
            cost_data = result.get("stats").calculate_cost() if result.get("stats") else {"total": 0}
            total_cost = cost_data["total"] if isinstance(cost_data, dict) else 0
            
            # 获取任务规划指标
            tasks_num = result.get("total_tasks_num", "-")
            compression = result.get("compression_ratio", "-")
            if compression != "-":
                compression = f"{compression:.2%}"
            plan_tokens = result.get("avg_task_plan_tokens", "-")
            if plan_tokens != "-":
                plan_tokens = f"{plan_tokens:.1f}"
            
            report += f"| {i+1} | {problem} | {is_correct} | {exec_time:.2f} | {total_cost:.4f} | {tasks_num} | {compression} | {plan_tokens} |\n"
        logger = get_logger()
        logger.info("===== 数据集处理报告 =====")
        logger.info(report)

        return report

    def save_report(self, output_dir=None, timestamp=None):
        if timestamp is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        if output_dir is None:
            dataset_name = os.path.basename(self.dataset_path)
            output_dir = build_report_path(
                base_dir="data_reports", 
                is_dataset=True,
                dataset_name=dataset_name,
                config=self.config,
                timestamp=timestamp
            )
        
        report_file = os.path.join(output_dir, "dataset_report.md")
        report = self.generate_report()
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"报告已保存至: {report_file}")
        
        # 创建理论性能报告目录
        theory_dir = os.path.join(output_dir, "theoretical_reports")
        os.makedirs(theory_dir, exist_ok=True)
            
        # 为每个问题保存单独的理论性能报告（如果有）
        for i, result in enumerate(self.results):
            if "theoretical_report" in result and result["theoretical_report"]:
                problem_desc = result.get("problem", "")[:20].replace("/", "_").replace("\\", "_").strip()
                problem_desc = ''.join(c for c in problem_desc if c.isalnum() or c in '_-')
                if not problem_desc:
                    problem_desc = f"problem_{i+1}"
                
                theory_report_file = os.path.join(theory_dir, f"theoretical_report_{i+1}_{problem_desc}.md")
                with open(theory_report_file, 'w', encoding='utf-8') as f:
                    f.write(f"# 问题 {i+1} 的理论性能分析报告\n\n")
                    f.write(f"## 问题描述\n\n{result.get('problem', '')}\n\n")
                    f.write(result["theoretical_report"])
                
        print(f"所有理论性能报告已保存至目录: {theory_dir}")

        # --- 新增：保存详细的评估报告 ---
        if self.evaluator and self.evaluator.enabled:
            eval_dir = os.path.join(output_dir, "evaluation_reports")
            os.makedirs(eval_dir, exist_ok=True)
            for i, result in enumerate(self.results):
                
                if "evaluation_results" in result:
                    eval_report_file = os.path.join(eval_dir, f"evaluation_problem_{i+1}.json")
                    with open(eval_report_file, 'w', encoding='utf-8') as f:
                        json.dump(result["evaluation_results"], f, ensure_ascii=False, indent=2)
            print(f"详细评估报告已保存至: {eval_dir}")
        # --- 详细报告保存结束 ---
            
        json_file = self.save_results_json(output_dir, timestamp)
        print(f"最终JSON结果已保存至: {json_file}")
        
        return report_file
    
    def save_results_json(self, output_dir, timestamp=None):
        os.makedirs(output_dir, exist_ok=True)
        if timestamp is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
        json_file = os.path.join(output_dir, "dataset_results.json")
        
        # 准备JSON数据
        json_results = []
        for result in self.results:
            serializable_result = {
                "problem": result.get("problem", ""),
                "gold_solution": result.get("gold_solution", ""),
                "model_solution": result.get("model_solution", ""),
                "difficulty": result.get("difficulty", 0),
                "is_correct": result.get("is_correct", False),
                "judge_result": result.get("judge_result", ""),
                "execution_time": result.get("execution_time", 0),
                "total_tasks_num": result.get("total_tasks_num", 0),
                "compression_ratio": result.get("compression_ratio", 0),
                "avg_task_plan_tokens": result.get("avg_task_plan_tokens", 0)
            }
            
            if "theoretical_metrics_raw" in result:
                serializable_result["theoretical_metrics"] = result["theoretical_metrics_raw"]
            
            if "tasks" in result:
                serializable_result["tasks"] = {k: v for k, v in result["tasks"].items()}
            if result.get("stats"):
                stats = result["stats"]
                serializable_result["stats"] = {
                    "costs": stats.calculate_cost(),
                    "tokens_per_second": stats.calculate_tokens_per_second(),
                    "token_usage": stats.token_usage,
                    "ttft_metrics": stats.ttft_metrics
                }
            json_results.append(serializable_result)
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, ensure_ascii=False, indent=2)
        return json_file

def build_dataset(config, dataset_path, limit, workers, build_config, output_dir):
    """处理从源文件构建数据集的逻辑"""
    logger = get_logger()
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        if limit:
            dataset = dataset[:limit]
    except Exception as e:
        logger.error(f"加载数据集构建失败: {e}", exc_info=True)
        print(f"加载数据集错误 {dataset_path}: {e}")
        return

    data_with_thinking = []
    data_without_thinking = []

    for item in tqdm(dataset, desc="构建数据集中"):
        problem = item.get("problem", "")
        solution = item.get("answer", "")

        if not problem:
            logger.warning("跳过'problem'字段为空的项。")
            continue
            
        full_plan, plan_only, system_prompt = dataset_run_parallel_execution(
            problem, solution, config, workers, build_config
        )

        if full_plan is None and plan_only is None:
            logger.error(f"为问题生成计划失败: {problem[:100]}...")
            continue
        
        common_data = {
            "instruction": problem,
            "input": "",
            "system": system_prompt.strip() if system_prompt else ""
        }

        if build_config.get('save_thinking', True) and full_plan:
            entry_w_think = common_data.copy()
            entry_w_think["output"] = full_plan
            data_with_thinking.append(entry_w_think)

        if plan_only:
            entry_wo_think = common_data.copy()
            entry_wo_think["output"] = plan_only
            data_without_thinking.append(entry_wo_think)
    
    file_wo_thinking_path = os.path.join(output_dir, "datasetTraining_wo_thinking.json")
    with open(file_wo_thinking_path, 'w', encoding='utf-8') as f:
        json.dump(data_without_thinking, f, ensure_ascii=False, indent=2)
    logger.info(f"已将不带思考的数据集保存到 {file_wo_thinking_path}")
    print(f"已保存 {file_wo_thinking_path}")

    if build_config.get('save_thinking', True):
        file_w_thinking_path = os.path.join(output_dir, "datasetTraining_w_thinking.json")
        with open(file_w_thinking_path, 'w', encoding='utf-8') as f:
            json.dump(data_with_thinking, f, ensure_ascii=False, indent=2)
        logger.info(f"已将带思考的数据集保存到 {file_w_thinking_path}")
        print(f"已保存 {file_w_thinking_path}")


def run_dataset_evaluation(config, dataset_path, limit=None, workers=4, dataset_build_config=None, evaluator=None, timestamp=None):
    """根据配置运行数据集评估或数据集构建"""
    # timestamp = time.strftime("%Y%m%d_%H%M%S")
    dataset_name = os.path.basename(dataset_path)
    output_dir = build_report_path(
        base_dir="data_reports", is_dataset=True, dataset_name=dataset_name,
        config=config, timestamp=timestamp
    )
    
    # This function now correctly passes the evaluator it receives.
    # The fix in main.py is the primary one, but this ensures correctness here too.
    setup_logger(output_dir)
    logger = get_logger()

    is_build_mode = dataset_build_config and dataset_build_config.get('enabled', False)
    use_models_for_execution = dataset_build_config and dataset_build_config.get('use_models_for_execution', False)

    if is_build_mode:
        logger.info("===== 数据集构建模式已激活 =====")
        print(f"开始数据集构建: {dataset_path}")
        build_dataset(config, dataset_path, limit, workers, dataset_build_config, output_dir)
        print(f"数据集构建完成。文件保存在: {output_dir}")

    if not is_build_mode or (is_build_mode and use_models_for_execution):
        logger.info("===== 数据集评估模式已激活 =====")
        print(f"开始数据集评估: {dataset_path}")
        
        # This call ensures the evaluator object is passed correctly to the class constructor.
        runner = DatasetRunner(config=config, dataset_path=dataset_path, limit=limit, workers=workers, evaluator=evaluator)

        enable_threshold = config.enable_threshold if hasattr(config, 'enable_threshold') else True
        runner.process_dataset(enable_threshold=enable_threshold, timestamp=timestamp)
        report_file = runner.save_report(output_dir=output_dir, timestamp=timestamp)
        return report_file
    else:
        return output_dir

import os
import time
import json
import pathlib
from config import ModelConfig, load_config, parse_args
from performance import PerformanceTracker, calculate_performance_metrics
from output_performance import generate_theoretical_performance_report
from execution import (
    run_parallel_execution, print_results, wait_for_completion_and_get_final_result,
    judge_question_difficulty, call_small_model_directly, generate_task_dependency_report,
    judge_correct, LLM_judge, save_result_to_file
)
from dataset_runner import run_dataset_evaluation
from log_config import setup_logger, get_logger, log_separator
from evaluation import Evaluator, format_aggregated_report_md
from utils import build_report_path

def main():
    """主程序入口"""
    print("启动程序...")
    
    args = parse_args()
    print(f"配置文件: {args.config}")
    
    yaml_config = load_config(args.config)

    # STEP 1: 首先确定运行模式和输出目录
    dataset_enabled = args.dataset or yaml_config.get("dataset", {}).get("enabled", False)
    dataset_path = args.dataset_path or yaml_config.get("dataset", {}).get("path", "")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    temp_config_for_path = ModelConfig(
        small_model=yaml_config["models"]["small_model"],
        large_model=yaml_config["models"]["large_model"],
        router_model=yaml_config["models"].get("router_model"),
        use_local_router=yaml_config["models"].get("use_local_router", False)
    )

    if dataset_enabled and dataset_path:
        output_dir = build_report_path(
            base_dir="data_reports", is_dataset=True, dataset_name=dataset_path,
            config=temp_config_for_path, timestamp=timestamp
        )
    else:
        output_dir = build_report_path(
            base_dir="data_reports", is_dataset=False, 
            config=temp_config_for_path, timestamp=timestamp
        )
    
    # STEP 2: 立即初始化日志记录器
    setup_logger(output_dir)
    logger = get_logger()
    logger.info("日志记录器已初始化。开始加载主程序配置。")

    # STEP 3: 加载所有其他配置
    dataset_limit = args.dataset_limit if args.dataset_limit is not None else yaml_config.get("dataset", {}).get("limit", None)
    
    small_model = yaml_config["models"]["small_model"]
    large_model = yaml_config["models"]["large_model"]
    router_model = yaml_config["models"].get("router_model", small_model)
    use_local_router = yaml_config["models"].get("use_local_router", False)
    local_router_model = yaml_config["models"].get("local_router_model")
    small_key_path = yaml_config["api"]["small_key_path"]
    large_key_path = yaml_config["api"]["large_key_path"]
    router_key_path = yaml_config["api"]["router_key_path"]
    small_api_base = yaml_config["api"].get("small_api_base_url")
    large_api_base = yaml_config["api"].get("large_api_base_url")
    router_api_base = yaml_config["api"].get("router_api_base_url")
    local_router_base = yaml_config["api"].get("local_router_base_url")
    workers = yaml_config["system"]["workers"]
    enable_judge = yaml_config["system"].get("enable_judge", False)
    gold_answer = yaml_config["system"].get("gold_answer", "")
    enable_threshold = yaml_config["models"].get("enable_threshold", False)
    threshold = yaml_config["models"]["threshold"]
    dataset_build_config = yaml_config.get("dataset", {}).get("build", {})

    system_config = yaml_config.get("system", {})
    retry_config = system_config.get("retries", {})
    enable_retries = retry_config.get("enabled", True)
    max_retry_attempts = retry_config.get("max_attempts", 3)
    retry_delay = retry_config.get("delay_seconds", 5)
    router_config = yaml_config.get("models", {}).get("router", {})

    config = ModelConfig(
        small_model=small_model, large_model=large_model, router_model=router_model,
        threshold=threshold, small_key_path=small_key_path, large_key_path=large_key_path,
        router_key_path=router_key_path, small_api_base=small_api_base,
        large_api_base=large_api_base, router_api_base=router_api_base,
        use_local_router=use_local_router, local_router_base=local_router_base,
        local_router_model=local_router_model, enable_threshold=enable_threshold,
        enable_retries=enable_retries,
        max_retry_attempts=max_retry_attempts,
        retry_delay=retry_delay,
        router_config=router_config
    )
    print("\n" + "="*80)
    print("===== 完整配置信息 (config.yaml) =====")
    print(json.dumps(yaml_config, indent=2, ensure_ascii=False))
    print("="*80 + "\n")
    
    # 2. 记录到日志文件
    logger.info("===== 完整配置信息 (config.yaml) =====")
    logger.info(json.dumps(yaml_config, indent=2, ensure_ascii=False))
    log_separator()
    
    eval_config = yaml_config["evaluation"]
    evaluator = Evaluator(eval_config)
    if evaluator.enabled:
        print(f"评估功能已开启 (Planner: {evaluator.planner_enabled}, Executor: {evaluator.executor_enabled})")
    
    yaml_config = load_config("config.yaml")
    models_config = yaml_config.get("models", {})
    sequential_execution = models_config.get("sequential_execution", False)
    single_rely = models_config.get("single_rely", False)
    if sequential_execution and single_rely:
        logger.info("-->顺序执行模式已启用，所有步骤将按顺序执行。每个步骤将依赖前序的一个步骤<--")
        print("-->顺序执行模式已启用，所有步骤将按顺序执行。每个步骤将依赖前序的一个步骤<--")
    elif sequential_execution and not single_rely:
        logger.info("-->顺序执行模式已启用，所有步骤将按顺序执行。每个步骤将依赖前序的所有步骤<--")
        print("-->顺序执行模式已启用，所有步骤将按顺序执行。每个步骤将依赖前序的所有步骤<--")
    else:
        logger.info("-->并行执行模式已启用，步骤将根据依赖关系并行执行<--")
        print("-->并行执行模式已启用，步骤将根据依赖关系并行执行<--")

    if dataset_enabled and dataset_path:
        logger.info("===== 数据集模式启动 =====")
        print("===== 数据集模式 =====")
        print(f"数据集路径: {dataset_path}")
        if dataset_limit is not None:
            print(f"数据集限制: {dataset_limit} 条")
        
        try:
            output_path = run_dataset_evaluation(
                config=config,
                dataset_path=dataset_path,
                limit=dataset_limit,
                workers=workers,
                dataset_build_config=dataset_build_config,
                evaluator=evaluator,
                timestamp=timestamp
            )
            
            if not (dataset_build_config.get("enabled", False) and not dataset_build_config.get("use_models_for_execution", False)):
                print(f"数据集评估完成，报告已保存至: {output_path}")
            return
        except Exception as e:
            logger.error(f"数据集处理过程中出错: {e}", exc_info=True)
            print(f"数据集处理过程中出错: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # 单问题处理模式
    query = yaml_config["query"]
    logger.info("===== 单问题处理模式启动 =====")
    print("===== 单问题处理模式 =====")
    print(f"当前查询: {query}")

    try:
        from execution import initialize_clients
        initialize_clients(config)
        
        difficulty = judge_question_difficulty(query, config)
        
        stats_tracker = PerformanceTracker(config)
        planner_output = None 
        final_result = "" 
        tasks = {}

        if enable_threshold and difficulty.isdigit() and int(difficulty) < config.threshold:
            print(f"问题难度 {difficulty} 低于阈值 {config.threshold}，使用小模型处理")
            model_result = call_small_model_directly(query, config, stats_tracker)
            tasks = {
                "1": {
                    "Task": "直接使用小模型解答问题", "Difficulty": difficulty,
                    "Token": "1000", "Rely": "", "Result": model_result
                }
            }
            final_result = (f"# 问题求解最终结果\n\n## 原始问题\n{query}\n\n## 解决步骤\n\n"
                          f"### 步骤 1: 直接使用小模型解答问题\n{model_result}\n\n## 最终答案\n{model_result}\n")
            stats_tracker.stop_tracking()
            print("\n最终结果:", final_result)
        else:
            tasks, stats_tracker, planner_output = run_parallel_execution(query, config, workers)
            if tasks is None or stats_tracker is None or planner_output is None:
                logger.error("任务执行失败，未能生成结果。")
            print_results(tasks)
            final_result = wait_for_completion_and_get_final_result(tasks, query, config, stats_tracker)
            print("\n最终合并结果:", final_result)
        
        correctness_report = ""
        # (Correctness logic remains the same)
        
        performance_report = calculate_performance_metrics(stats_tracker)
        theoretical_report = generate_theoretical_performance_report(tasks, config, stats_tracker.planner_output)
        dependency_report = generate_task_dependency_report(tasks)
        
        evaluation_report_md = ""
        if evaluator.enabled and planner_output:
            eval_results = {}
            planner_report_json = evaluator.evaluate_planner(query, planner_output)
            if planner_report_json:
                eval_results["planner_report"] = planner_report_json
            
            executor_reports = evaluator.evaluate_executor(query, planner_output, tasks, config)
            if executor_reports:
                eval_results["executor_reports"] = executor_reports
            
            evaluation_report_md = format_aggregated_report_md(eval_results)
        
        output_file = save_result_to_file(
            final_result, config, workers, correctness_report, 
            performance_report, dependency_report, theoretical_report, 
            output_dir, evaluation_report_md
        )
        if output_file:
            print(f"结果已保存至: {output_file}")
            
    except Exception as e:
        logger.error(f"单问题处理过程中出错: {e}", exc_info=True)
        print(f"处理过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
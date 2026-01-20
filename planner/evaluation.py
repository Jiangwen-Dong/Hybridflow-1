import json
import os
import re
from openai import OpenAI
from config import ModelConfig
# 导入日志配置模块
from log_config import get_logger, log_separator

class Evaluator:
    """
    Manages the evaluation of Planner and Executor models.
    """
    def __init__(self, config):
        """
        Initializes the Evaluator with a dedicated model configuration.
        """
        self.config = config
        self.enabled = config.get("enabled")
        self.planner_enabled = config.get("planner_enabled")
        self.executor_enabled = config.get("executor_enabled")

        if self.enabled:
            self.eval_model_name = config.get("model")
            key_path = config.get("key_path")
            api_base = config.get("api_base_url")
            
            try:
                with open(key_path, 'r') as f:
                    api_key = f.read().strip()
                
                self.client = OpenAI(api_key=api_key, base_url=api_base)
            except Exception as e:
                error_message = f"评估器初始化失败: {e}. 请检查配置文件中的 'key_path': '{key_path}' 是否正确。"
                try:
                    logger = get_logger()
                    logger.error(error_message)
                except Exception:
                    pass 

    def _load_prompt_template(self, file_path):
        """Loads a prompt template from a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            print(f"Evaluation prompt file not found: {file_path}")
            return None

    def _call_eval_model(self, system_prompt, user_prompt):
        """Calls the evaluation model and returns the parsed JSON response."""
        if not self.enabled or not self.client:
            return None
        
        logger = get_logger()
        logger.info("----------正在调用评估模型----------")
        logger.info(f"发送给评估模型的 Prompt:\nSystem: {system_prompt}\nUser: {user_prompt}\n")
        
        try:
            response = self.client.chat.completions.create(
                model=self.eval_model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            logger.info("成功收到评估模型的响应。")
            logger.info(f"评估模型的完整响应: \n{content}")
            log_separator()
            return json.loads(content)
        except Exception as e:
            error_msg = f"调用评估模型时出错: {e}"
            print(error_msg)
            logger.error(error_msg)
            return {"error": str(e)}

    def evaluate_planner(self, question, planner_output):
        if not self.planner_enabled:
            return None
        system_prompt = self._load_prompt_template("evaluationPrompt/PlannerEvaluation.txt")
        if not system_prompt:
            logger = get_logger()
            logger.error("无法加载规划器评估的系统提示模板。")
            return None
        user_prompt = f"**Question:**\n{question}\n\n**Plan:**\n```xml\n{planner_output}\n```"
        print("\n--- 正在评估规划器 (Planner) ---")
        report = self._call_eval_model(system_prompt, user_prompt)
        print("规划器评估完成。")
        return report

    def evaluate_executor(self, question, planner_output, tasks, model_config):
        if not self.executor_enabled:
            return None
        system_prompt = self._load_prompt_template("evaluationPrompt/ExecutorEvaluation.txt")
        if not system_prompt:
            logger = get_logger()
            logger.error("无法加载执行器评估的系统提示模板。")
            return None
        executor_reports = {}
        sorted_tasks = sorted(tasks.items(), key=lambda x: int(x[0]))
        print("\n--- 正在评估执行器 (Executors) ---")
        for step_id, task_details in sorted_tasks:
            difficulty = int(task_details.get('Difficulty', 0))
            model_used = model_config.large_model if difficulty >= model_config.threshold else model_config.small_model
            context_provided = ""
            rely_ids = task_details.get('Rely', '')
            if rely_ids:
                context_lines = ["**Context Provided (Results from prior steps):**"]
                for rely_id in rely_ids.split(','):
                    if rely_id in tasks and 'Result' in tasks[rely_id]:
                        prev_task = tasks[rely_id]
                        context_lines.append(f"- Task {rely_id} ({prev_task.get('Task', '')}): {prev_task.get('Result', '')}")
                context_provided = "\n".join(context_lines)
            user_prompt = (f"**Original Question:**\n{question}\n\n"
                           f"**Complete Plan:**\n```xml\n{planner_output}\n```\n\n"
                           f"**Sub-task Assigned to Executor:**\n- Step ID: {step_id}\n"
                           f"- Task: {task_details.get('Task', '')}\n\n"
                           f"{context_provided}\n\n"
                           f"**Executor's Response:**\n{task_details.get('Result', '')}")
            print(f"正在评估步骤 {step_id} (模型: {model_used})...")
            report = self._call_eval_model(system_prompt, user_prompt)
            executor_reports[step_id] = {"model_used": model_used, "report": report}
        print("执行器评估完成。")
        return executor_reports

def aggregate_dataset_reports(results):
    """
    Aggregates evaluation reports from a dataset run to calculate average scores.
    MODIFIED: Now groups executor scores by model name.
    """
    planner_scores = []
    # MODIFICATION: Use a dictionary to store scores for each model
    executor_scores_by_model = {}

    for result in results:
        eval_data = result.get("evaluation_results", {})
        if eval_data.get("planner_report"):
            report = eval_data["planner_report"].get("evaluationReport", {})
            # Use a helper to safely parse score, defaulting to None if not a valid number
            def safe_int(score):
                try: return int(score)
                except (ValueError, TypeError): return None

            scores = {key: safe_int(val.get('score')) for key, val in report.items() if isinstance(val, dict)}
            # Filter out entries where score could not be parsed
            valid_scores = {k: v for k, v in scores.items() if v is not None}
            if valid_scores:
                planner_scores.append(valid_scores)

        if eval_data.get("executor_reports"):
            for step_id, data in eval_data["executor_reports"].items():
                model_name = data.get("model_used")
                report = data.get("report", {}).get("evaluationReport", {})
                
                if model_name and report:
                    scores = {key: safe_int(val.get('score')) for key, val in report.items() if isinstance(val, dict)}
                    valid_scores = {k: v for k, v in scores.items() if v is not None}
                    
                    if valid_scores:
                        if model_name not in executor_scores_by_model:
                            executor_scores_by_model[model_name] = []
                        executor_scores_by_model[model_name].append(valid_scores)

    def calculate_averages(score_list):
        if not score_list:
            return {}
        
        # Aggregate all keys present in the list of score dicts
        all_keys = set()
        for s in score_list:
            all_keys.update(s.keys())

        avg = {}
        for key in all_keys:
            # Sum scores only from dicts that contain the key
            sum_of_scores = sum(s.get(key, 0) for s in score_list)
            count = sum(1 for s in score_list if key in s)
            if count > 0:
                avg[key] = sum_of_scores / count
        return avg

    avg_planner = calculate_averages(planner_scores)
    
    # MODIFICATION: Calculate averages for each model
    avg_executors = {}
    for model_name, score_list in executor_scores_by_model.items():
        avg_executors[model_name] = calculate_averages(score_list)
    
    return {"planner": avg_planner, "executors": avg_executors}

def format_aggregated_report_md(avg_scores):
    """
    Formats the aggregated scores into a Markdown section.
    MODIFIED: Now creates a separate table for each executor model.
    """
    if not avg_scores:
        return ""
    
    md = "\n## 平均评估分数\n"
    
    if avg_scores.get("planner"):
        md += "\n### 规划器平均分数\n"
        md += "| 维度 | 平均分 (满分5分) |\n"
        md += "| --- | --- |\n"
        # Sort items for consistent order
        for key, score in sorted(avg_scores["planner"].items()):
            # Clean up the key for better display
            display_key = ''.join(' ' + char if char.isupper() else char for char in key).replace('_', ' ').strip().title()
            md += f"| {display_key} | {score:.2f} |\n"
            
    # MODIFICATION: Handle the new 'executors' dictionary structure
    if avg_scores.get("executors"):
        md += "\n### 执行器平均分数\n"
        # Sort by model name for consistent order
        for model_name, scores in sorted(avg_scores["executors"].items()):
            md += f"\n#### 模型: `{model_name}`\n"
            md += "| 维度 | 平均分 (满分5分) |\n"
            md += "| --- | --- |\n"
            # Sort items for consistent order
            for key, score in sorted(scores.items()):
                display_key = ''.join(' ' + char if char.isupper() else char for char in key).replace('_', ' ').strip().title()
                md += f"| {display_key} | {score:.2f} |\n"
            
    return md
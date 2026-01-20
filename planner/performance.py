import time
from api_pricing import get_model_pricing

class PerformanceTracker:
    """性能跟踪器类，用于跟踪模型使用情况和成本"""
    
    def __init__(self, model_config=None):
        """初始化性能跟踪器
        
        参数:
            model_config: 可选，模型配置对象，用于获取最新的价格费率
        """
        self.start_time = time.time()
        self.end_time = None
        
        # 规划器(Planner)输出信息
        self.planner_output = None

        self.total_router_latency = 0.0

        # 首个令牌响应时间统计
        self.ttft_metrics = {
            "small_model": [],
            "large_model": [],
            "router_model": [],  # 添加路由模型的TTFT统计
            "total": []
        }
        
        # Token使用统计
        self.token_usage = {
            "small_model": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            },
            "large_model": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            },
            "router_model": {  # 添加路由模型的token统计
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            },
            "router_internal": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            },
            "total": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
        
        # 成本估算 (美元/百万tokens)
        if model_config and hasattr(model_config, 'get_model_cost_rates'):
            # 如果提供了模型配置，使用其费率
            self.cost_rates = model_config.get_model_cost_rates()
        else:
            self.cost_rates = {
                "small_model": get_model_pricing("llama-3-8b-instruct"),
                "large_model": get_model_pricing("gpt-4o"),
                "router_model": get_model_pricing("claude-3-5-sonnet")
            }

        self.cost_rates["router_internal"] = {
            "prompt": 0.0,
            "completion": 0.0
        }
        # print("DEBUG: 使用的模型费率:", self.cost_rates)
        
    
    def update_token_usage(self, model_type, prompt_tokens, completion_tokens):
        """更新token使用情况
        
        参数:
            model_type: 模型类型 ("small_model" 或 "large_model")
            prompt_tokens: 输入token数量
            completion_tokens: 输出token数量
        """
        # 更新指定模型的统计
        if model_type not in self.token_usage:
            print(f"警告: 未知的模型类型 '{model_type}'，无法更新 token usage。")
            model_type = "small_model" # 默认归入 small
        self.token_usage[model_type]["prompt_tokens"] += prompt_tokens
        self.token_usage[model_type]["completion_tokens"] += completion_tokens
        self.token_usage[model_type]["total_tokens"] += prompt_tokens + completion_tokens
        
        # 更新总统计
        self.token_usage["total"]["prompt_tokens"] += prompt_tokens
        self.token_usage["total"]["completion_tokens"] += completion_tokens
        self.token_usage["total"]["total_tokens"] += prompt_tokens + completion_tokens
        
    def update_ttft(self, model_type, ttft):
        """更新首个令牌响应时间 (Time to First Token)
        
        参数:
            model_type: 模型类型 ("small_model" 或 "large_model")
            ttft: 首个令牌响应时间（秒）
        """
        if model_type not in self.ttft_metrics:
            print(f"警告: 未知的模型类型 '{model_type}'，无法更新 TTFT。")
            model_type = "small_model" # 默认归入 small

        if ttft is not None:
            # 更新指定模型的统计
            self.ttft_metrics[model_type].append(ttft)
            # 更新总统计
            self.ttft_metrics["total"].append(ttft)
    
    def update_router_latency(self, latency_s: float):
        """累积智能路由决策所花费的时间"""
        if latency_s:
            self.total_router_latency += latency_s

    def get_total_tokens(self) -> int:
        """获取所有模型到目前为止消耗的总 token 数"""
        return self.token_usage["total"]["total_tokens"]

    def save_planner_output(self, prompt_tokens, completion_tokens, ttft):
        """保存planner输出的性能信息
        
        参数:
            prompt_tokens: 输入token数量
            completion_tokens: 输出token数量
            ttft: 首个令牌响应时间（秒）
        """
        self.planner_output = {
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'ttft': ttft,
            'total_tokens': prompt_tokens + completion_tokens
        }
    
    def stop_tracking(self):
        """停止性能跟踪"""
        self.end_time = time.time()
    
    def calculate_cost(self):
        """计算总成本
        
        返回:
            总成本（美元）
        """
        # API价格通常以美元/百万tokens为单位，因此需要除以1,000,000
        small_model_cost = (
            (self.token_usage["small_model"]["prompt_tokens"] / 1000000) * self.cost_rates["small_model"]["prompt"] +
            (self.token_usage["small_model"]["completion_tokens"] / 1000000) * self.cost_rates["small_model"]["completion"]
        )
        
        large_model_cost = (
            (self.token_usage["large_model"]["prompt_tokens"] / 1000000) * self.cost_rates["large_model"]["prompt"] +
            (self.token_usage["large_model"]["completion_tokens"] / 1000000) * self.cost_rates["large_model"]["completion"]
        )
        
        router_model_cost = (
            (self.token_usage["router_model"]["prompt_tokens"] / 1000000) * self.cost_rates["router_model"]["prompt"] +
            (self.token_usage["router_model"]["completion_tokens"] / 1000000) * self.cost_rates["router_model"]["completion"]
        )

        router_internal_cost = (
            (self.token_usage["router_internal"]["prompt_tokens"] / 1000000) * self.cost_rates["router_internal"]["prompt"] +
            (self.token_usage["router_internal"]["completion_tokens"] / 1000000) * self.cost_rates["router_internal"]["completion"]
        )

        total_cost = small_model_cost + large_model_cost + router_model_cost + router_internal_cost

        return {
            "small_model": small_model_cost,
            "large_model": large_model_cost,
            "router_model": router_model_cost,
            "router_internal": router_internal_cost,
            "total": total_cost
        }
    
    def get_elapsed_time(self):
        """获取总耗时
        
        返回:
            耗时（秒）
        """
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time
    
    def calculate_tokens_per_second(self):
        """计算每秒生成的token数量"""
        elapsed_time = self.get_elapsed_time()
        if elapsed_time <= 0:
            return {
                "small_model": 0,
                "large_model": 0,
                "router_model": 0,
                "router_internal": 0, 
                "total": 0
            }
        
        small_tps = self.token_usage["small_model"]["completion_tokens"] / elapsed_time
        large_tps = self.token_usage["large_model"]["completion_tokens"] / elapsed_time
        router_tps = self.token_usage["router_model"]["completion_tokens"] / elapsed_time
        router_internal_tps = self.token_usage["router_internal"]["completion_tokens"] / elapsed_time
        total_tps = self.token_usage["total"]["completion_tokens"] / elapsed_time
        
        return {
            "small_model": small_tps,
            "large_model": large_tps,
            "router_model": router_tps,
            "router_internal": router_internal_tps, 
            "total": total_tps
        }
    
    def format_performance_report(self):
        """格式化性能报告
        
        返回:
            性能报告文本
        """
        costs = self.calculate_cost()
        elapsed_time = self.get_elapsed_time()
        tokens_per_second = self.calculate_tokens_per_second()
        
        report = "# 性能统计报告\n\n"
        report += f"## 总执行时间\n{elapsed_time:.3f} 秒\n\n"
        
        # 计算平均TTFT
        def calc_avg_ttft(ttft_list):
            return sum(ttft_list) / len(ttft_list) if ttft_list else 0
            
        small_ttft = self.ttft_metrics["small_model"]
        large_ttft = self.ttft_metrics["large_model"]
        all_ttft = self.ttft_metrics["total"]

        def calc_total_ttft(ttft_list):
            return sum(ttft_list) if ttft_list else 0
        
        report += f"## 去除ttft的总执行时间\n{elapsed_time - calc_total_ttft(all_ttft):.3f} 秒\n\n"
        report += f"## 智能路由决策耗时\n"
        report += f"**总耗时:** {self.total_router_latency:.3f} 秒\n\n"
        # 首个令牌响应时间报告
        report += "## 首个令牌响应时间 (TTFT)\n\n"

        report += "### 小模型\n"
        if small_ttft:
            report += f"- 平均首个令牌响应时间: {calc_avg_ttft(small_ttft):.3f} 秒\n"
            report += f"- 最短响应时间: {min(small_ttft):.3f} 秒\n"
            report += f"- 最长响应时间: {max(small_ttft):.3f} 秒\n"
            report += f"- 响应次数: {len(small_ttft)}\n\n"
        else:
            report += "- 无数据\n\n"
            
        report += "### 大模型\n"
        if large_ttft:
            report += f"- 平均首个令牌响应时间: {calc_avg_ttft(large_ttft):.3f} 秒\n"
            report += f"- 最短响应时间: {min(large_ttft):.3f} 秒\n"
            report += f"- 最长响应时间: {max(large_ttft):.3f} 秒\n"
            report += f"- 响应次数: {len(large_ttft)}\n\n"
        else:
            report += "- 无数据\n\n"
            
        report += "### 总计\n"
        if all_ttft:
            report += f"- 平均首个令牌响应时间: {calc_avg_ttft(all_ttft):.3f} 秒\n"
            report += f"- 最短响应时间: {min(all_ttft):.3f} 秒\n"
            report += f"- 最长响应时间: {max(all_ttft):.3f} 秒\n"
            report += f"- 响应总次数: {len(all_ttft)}\n\n"
        else:
            report += "- 无数据\n\n"
        
        report += "## Token 使用情况\n\n"
        report += "### 小模型\n"
        report += f"- 输入 Tokens: {self.token_usage['small_model']['prompt_tokens']}\n"
        report += f"- 输出 Tokens: {self.token_usage['small_model']['completion_tokens']}\n"
        report += f"- 总 Tokens: {self.token_usage['small_model']['total_tokens']}\n\n"
        
        report += "### 大模型\n"
        report += f"- 输入 Tokens: {self.token_usage['large_model']['prompt_tokens']}\n"
        report += f"- 输出 Tokens: {self.token_usage['large_model']['completion_tokens']}\n"
        report += f"- 总 Tokens: {self.token_usage['large_model']['total_tokens']}\n\n"
        
        report += "### 路由模型\n"
        report += f"- 输入 Tokens: {self.token_usage['router_model']['prompt_tokens']}\n"
        report += f"- 输出 Tokens: {self.token_usage['router_model']['completion_tokens']}\n"
        report += f"- 总 Tokens: {self.token_usage['router_model']['total_tokens']}\n\n"
        
        report += "### 智能路由 (内部)\n"
        report += f"- 输入 Tokens: {self.token_usage['router_internal']['prompt_tokens']}\n"
        report += f"- 输出 Tokens: {self.token_usage['router_internal']['completion_tokens']}\n"
        report += f"- 总 Tokens: {self.token_usage['router_internal']['total_tokens']}\n\n"

        report += "### 总计\n"
        report += f"- 输入 Tokens: {self.token_usage['total']['prompt_tokens']}\n"
        report += f"- 输出 Tokens: {self.token_usage['total']['completion_tokens']}\n"
        report += f"- 总 Tokens: {self.token_usage['total']['total_tokens']}\n\n"
        
        report += "## 生成速度\n\n"
        report += f"- 小模型每秒生成token数: {tokens_per_second['small_model']:.2f} tokens/s\n"
        report += f"- 大模型每秒生成token数: {tokens_per_second['large_model']:.2f} tokens/s\n"
        report += f"- 路由模型每秒生成token数: {tokens_per_second['router_model']:.2f} tokens/s\n"
        report += f"- 平均每秒生成token数: {tokens_per_second['total']:.2f} tokens/s\n\n"
        
        report += "## 成本估算\n\n"
        report += f"- 小模型成本: ${costs['small_model']:.4f}\n"
        report += f"- 大模型成本: ${costs['large_model']:.4f}\n"
        report += f"- 路由模型成本: ${costs['router_model']:.4f}\n"
        report += f"- 总成本: ${costs['total']:.4f}\n"
        
        return report


def calculate_performance_metrics(stats_tracker):
    """计算性能指标
    
    参数:
        stats_tracker: 性能统计跟踪器
    
    返回:
        性能报告文本
    """
    stats_tracker.stop_tracking()
    return stats_tracker.format_performance_report()

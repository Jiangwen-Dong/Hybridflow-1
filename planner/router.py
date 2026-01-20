# router_logic.py
import time
import random
import logging
from typing import List, Dict, Any, Tuple

import torch
from openai import OpenAI

# --- 假设 RouterNet 和 RouterConfig 在 router_train.py 中 ---
try:
    # 您需要确保这个文件存在且可被导入
    from router_train import RouterNet, RouterConfig
except ImportError:
    print("警告: 'router_train.py' 未找到或无法导入。")
    print("智能路由功能将无法使用。请确保 RouterNet 和 RouterConfig 已定义。")
    
    # 定义假的占位符，以防导入失败但仍继续运行
    class RouterConfig:
        def __init__(self):
            self.tau = 1.0
    class RouterNet(torch.nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg
            print("警告: 正在使用假的 RouterNet 占位符。")
        def forward(self, embedding, meta, rB_t, rL_t):
            return torch.tensor([0.0])

# --- 用于缓存模型和客户端的全局变量 ---
ROUTER_MODEL = None
ROUTER_CLIENT = None

def pre_evaluate_step(
    question: str, 
    subproblem: dict, 
    cumulative_tokens: int, 
    cumulative_latency: float, 
    all_steps: List[Dict[str, Any]],
    router_config: Dict[str, Any],
    logger: logging.Logger
) -> Tuple[bool, Dict[str, Any]]:
    """
    决定一个子任务是路由到云端（大模型）还是边缘（小模型）。
    
    参数:
        question: 原始问题
        subproblem: 当前步骤的任务字典 (来自 tasks[step_id])
        cumulative_tokens: 到目前为止的总 token 消耗
        cumulative_latency: 到目前为止的总 latency 消耗
        all_steps: 包含所有步骤字典的列表 (用于规则判断，例如最后一步)
        router_config: 包含所有路由参数的字典 (来自 config.yaml)
        logger: 日志记录器
        
    返回:
        (use_cloud: bool, router_metrics: dict)
        use_cloud 为 True 表示使用大模型 (Cloud)，False 表示使用小模型 (Edge)
    """
    global ROUTER_MODEL, ROUTER_CLIENT

    # --- 从传入的 router_config 中获取配置 ---
    ckpt_path = router_config.get("ckpt_path", "/home/jwdong/demo2/router/router_trained_vllm.pt")
    embed_model = router_config.get("embed_model", "Qwen/Qwen3-Embedding-0.6B")
    vllm_url = router_config.get("vllm_url", "http://localhost:8001/v1")
    
    # 预算和超参数
    B_max = router_config.get("token_budget_max", 2048.0)
    L_max = router_config.get("latency_budget_max", 20.0)
    delta0 = router_config.get("delta0", 0.0)
    alpha_B = router_config.get("alpha_B", 0.5)
    alpha_L = router_config.get("alpha_L", 0.5)
    
    cfg = RouterConfig()
    cfg.tau = router_config.get("tau", 0.1) # Sigmoid 温度

    # --- 初始化模型和客户端 (仅一次) ---
    try:
        if ROUTER_MODEL is None:
            ROUTER_MODEL = RouterNet(cfg)
            ROUTER_MODEL.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
            ROUTER_MODEL.eval()
            logger.info(f"RouterNet 模型从 {ckpt_path} 加载成功。")
        
        if ROUTER_CLIENT is None:
            ROUTER_CLIENT = OpenAI(base_url=vllm_url, api_key="dummy") # 假设 vLLM key 是 dummy
            logger.info(f"Router 嵌入客户端初始化，指向 {vllm_url}")
            
    except Exception as e:
        logger.error(f"初始化路由模型或客户端失败: {e}")
        # 失败时，默认路由到大模型 (安全起见)
        return True, {"error": str(e), "latency_s": 0.0, "model_type": "router_internal"}

    start_t = time.time()
    
    try:
        # --- 1. 获取嵌入 ---
        # 注意: 您的框架使用 'Task' 字段, 而不是 'desc'
        task_description = subproblem.get("Task", "") 
        text_to_embed = question + task_description
        
        resp = ROUTER_CLIENT.embeddings.create(model=embed_model, input=text_to_embed)
        embedding = torch.tensor(resp.data[0].embedding, dtype=torch.float32).unsqueeze(0)

        # --- 2. 准备 Meta-data ---
        # 注意: 您的框架使用 'Rely' 字段
        dep_depth = len(subproblem.get("Rely", "").split(',')) if subproblem.get("Rely") else 0
        tok_len = len(text_to_embed.split()) # 简单的 token 计数
        complexity = min(tok_len / 100.0, 1.0) # 假设的复杂度计算
        meta = torch.tensor([[dep_depth, tok_len, complexity, 0, 0, 0, 0, 0]], dtype=torch.float32)

        # --- 3. 准备资源使用率 ---
        r_B = cumulative_tokens / B_max
        r_L = cumulative_latency / L_max
        rB_t = torch.tensor([r_B], dtype=torch.float32)
        rL_t = torch.tensor([r_L], dtype=torch.float32)

        # --- 4. 模型推理 ---
        with torch.no_grad():
            s = torch.sigmoid(ROUTER_MODEL(embedding, meta, rB_t, rL_t)).item()
        
        # --- 5. 路由决策 (自适应分配) ---
        δ_t = delta0 + alpha_B * (r_B) + alpha_L * (r_L)
        if δ_t <= 0:
            δ_t = 0
        elif δ_t >= 1:
            δ_t = 1
        p_cloud = torch.sigmoid(torch.tensor(s - δ_t) / cfg.tau).item()
        use_cloud = (random.random() < p_cloud)

        # --- 6. 规则干预 (来自 hybridflow_runner.py) ---
        # 找到最后一步
        last_step_id = str(len(all_steps))
        current_step_id = subproblem.get("ID")
        
        if current_step_id == last_step_id:
             use_cloud = True # 强制最后一步使用大模型
             logger.info(f"[Router] 步骤 {current_step_id} 是最后一步，强制使用 CLOUD。")

        if current_step_id == "1":
             use_cloud = True # 强制第一步(Explain)使用大模型
             logger.info(f"[Router] 步骤 {current_step_id} 是第一步，强制使用 CLOUD。")
        
        end_t = time.time()
        latency = end_t - start_t
        
        # --- 7. 构造指标 ---
        pre_metrics = {
            "ttft_s": 0.0, # 嵌入通常没有TTFT
            "latency_s": latency,
            "n_input_tokens": tok_len, # 粗略估计
            "n_output_tokens": embedding.shape[1], # 嵌入维度
            "offload_cloud": use_cloud,
            "model_type": "router_internal", # 用于性能跟踪的特殊类型
            "score": s,
            "delta_t": δ_t,
            "p_cloud": p_cloud
        }

        if use_cloud:
            logger.info(f"[Router] 步骤 '{current_step_id}' → CLOUD (大模型) | s={s:.3f} δ_t={δ_t:.3f} p_cloud={p_cloud:.3f}")
        else:
            logger.info(f"[Router] 步骤 '{current_step_id}' → EDGE (小模型) | s={s:.3f} δ_t={δ_t:.3f} p_cloud={p_cloud:.3f}")

        return use_cloud, pre_metrics

    except Exception as e:
        logger.error(f"[Router] 路由步骤 '{subproblem.get('ID')}' 失败: {e}", exc_info=True)
        # 失败时，默认路由到大模型 (安全起见)
        return True, {"error": str(e), "latency_s": time.time() - start_t, "model_type": "router_internal"}

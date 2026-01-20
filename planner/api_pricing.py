# Update 7/10/2025 from https://openrouter.ai/models

from typing import Dict, Any
from log_config import setup_logger, get_logger, log_separator


def get_model_pricing(model_name: str) -> Dict[str, float]:
    """
    根据模型名称返回其价格费率（美元/百万tokens）
    
    参数:
        model_name: 模型名称（字符串或包含模型名称的字符串）
    
    返回:
        包含输入和输出价格的字典 {'prompt': float, 'completion': float}
    """
    # 标准化模型名称为小写以便匹配
    model_name_lower = model_name.lower()
    
    # 移除可能存在的路径或平台前缀（如 meta-llama/、deepseek/、openai/ 等）
    if "/" in model_name_lower:
        model_name_lower = model_name_lower.split("/")[1]
    
    # Claude模型价格
    if "claude-3-5" in model_name_lower or "claude-3.5" in model_name_lower or "claude-3-5-sonnet" in model_name_lower:
        return {"prompt": 3.0, "completion": 15.0}
    elif "claude-3-7" in model_name_lower or "claude-3.7" in model_name_lower or "claude-3-7-sonnet" in model_name_lower:
        return {"prompt": 3.0, "completion": 15.0}
    
    # Gemini模型价格 
    elif "gemini-2.5-flash-thinking" in model_name_lower:
        return {"prompt": 0.30, "completion": 2.50}
    elif "gemini-2.5-pro" in model_name_lower:
        return {"prompt": 1.25, "completion": 10.0}

    # OpenAI模型价格
    elif "gpt-4o" in model_name_lower:
        return {"prompt": 2.5, "completion": 10.0}
    elif "gpt-5" in model_name_lower:
        return {"prompt": 1.25, "completion": 10.0}
    elif "gpt-4.1-mini" in model_name_lower:
        return {"prompt": 0.4, "completion": 1.6}
    elif "gpt-4.1" in model_name_lower:
        return {"prompt": 2.0, "completion": 8.0}

    # DeepSeek系列
    elif "deepseek-r1" in model_name_lower:
        return {"prompt": 0.272, "completion": 0.272}
    elif "deepseek-chat" in model_name_lower:
        return {"prompt": 0.25, "completion": 0.85}
    elif "deepseek-reasoner" in model_name_lower:
        return {"prompt": 0.272, "completion": 0.272}
    
    # Grok模型价格
    elif "grok-4" in model_name_lower:
        return {"prompt": 3.0, "completion": 15.0}
    
    # Llama系列
    elif "llama3-8b" in model_name_lower or "llama-3-8b" in model_name_lower or "llama3.1-8b" in model_name_lower or "llama-3.1-8b" in model_name_lower:
        return {"prompt": 0.03, "completion": 0.06}
    elif "llama-3.2-3b" in model_name_lower or "llama3.2-3b" in model_name_lower:
        return {"prompt": 0.00, "completion": 0.00}
    elif "llama-3.3-70b" in model_name_lower or "llama3.3-70b" in model_name_lower:
        return {"prompt": 0.012, "completion": 0.036}
    elif "llama-3.2-1b" in model_name_lower or "llama3.2-1b" in model_name_lower:
        return {"prompt": 0.00, "completion": 0.00}

    # Qwen系列
    elif "qwen3-4b" in model_name_lower or "qwen-4b" in model_name_lower:
        return {"prompt": 0.00, "completion": 0.00}
    elif "qwen2.5-3b" in model_name_lower:
        return {"prompt": 0.00, "completion": 0.00}
    elif "qwen3-1.7b" in model_name_lower:
        return {"prompt": 0.00, "completion": 0.00}
    elif "qwen3-235b-a22b" in model_name_lower:
        return {"prompt": 0.10, "completion": 0.39}
    elif "qwen3-1.7b" in model_name_lower:
        return {"prompt": 0.00, "completion": 0.00}
    elif "qwen3-0.6b" in model_name_lower:
        return {"prompt": 0.00, "completion": 0.00}

    # 本地模型
    elif "saves" in model_name_lower or "sft" in model_name_lower:
        return {"prompt": 0.0, "completion": 0.0}

    # 默认价格（当无法识别模型时使用）
    else:
        print(f"警告: 未识别的模型 '{model_name}'，使用默认价格。请在 api_pricing.py 中添加此模型的价格信息。")
        logger = get_logger()
        logger.warning(f"未识别的模型 '{model_name}'，使用默认价格: prompt:0.0，completion:0.0。请在 api_pricing.py 中添加此模型的价格信息。")
        return {"prompt": 0.0, "completion": 0.0}

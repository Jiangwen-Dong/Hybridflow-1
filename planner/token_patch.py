import tiktoken
import transformers
import os
import sys

_tiktoken_encoder = None

def get_tiktoken_encoder():
    """获取 tiktoken 编码器 (cl100k_base)"""
    global _tiktoken_encoder
    if _tiktoken_encoder is None:
        try:
            _tiktoken_encoder = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            print(f"错误: 无法加载 tiktoken 编码 'cl100k_base'。")
            print(f"请确保已安装 tiktoken: pip install tiktoken")
            print(f"详细错误: {e}")
            sys.exit(1)
    return _tiktoken_encoder

def count_tiktoken_tokens(text: str) -> int:
    """使用 cl100k_base (GPT-4/4.1) 计算 token 数量"""
    encoder = get_tiktoken_encoder()
    return len(encoder.encode(text))

_deepseek_tokenizer = None

def get_deepseek_tokenizer():
    """获取 DeepSeek tokenizer 实例（如果存在）"""
    global _deepseek_tokenizer
    if _deepseek_tokenizer is None:
        try:
            # 假设 deepseek_v3_tokenizer 文件夹在同一目录
            chat_tokenizer_dir = os.path.join(os.path.dirname(__file__), 'deepseek_v3_tokenizer')
            if not os.path.isdir(chat_tokenizer_dir):
                raise FileNotFoundError("DeepSeek tokenizer 文件夹未找到")
                
            _deepseek_tokenizer = transformers.AutoTokenizer.from_pretrained(
                chat_tokenizer_dir,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"警告: 无法加载 DeepSeek tokenizer。DeepSeek 模型的 Token 计数将回退到 tiktoken。")
            print(f"详细信息: {e}")
            _deepseek_tokenizer = "failed" 
            
    return _deepseek_tokenizer

def count_deepseek_tokens(text: str) -> int:
    """使用 DeepSeek tokenizer 计算 token 数量"""
    tokenizer = get_deepseek_tokenizer()
    
    if tokenizer == "failed" or tokenizer is None:
        return count_tiktoken_tokens(text)
        
    return len(tokenizer.encode(text))


def count_tokens_for_model(text: str, model_name: str) -> int:
    """
    根据模型名称选择正确的 tokenizer 来计算 token 数量。
    """
    if text is None:
        return 0
    if model_name is None:
        model_name = ""
            
    model_lower = model_name.lower()
    
    if "gpt" in model_lower or "claude" in model_lower or "qwen" in model_lower:
        return count_tiktoken_tokens(text)
    
    elif "deepseek" in model_lower:
        return count_deepseek_tokens(text)
        
    else:
        return count_tiktoken_tokens(text)
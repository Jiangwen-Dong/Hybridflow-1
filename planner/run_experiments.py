import yaml
import subprocess
import os
import shutil
import time
from log_config import setup_logger, get_logger
from tqdm import tqdm

# --- 1. 配置区域 ---
# 在这里定义您要运行的所有实验
# 每个字典代表一次独立的实验配置
# EXPERIMENTS_TO_RUN = [
#     {
#         'name': 'Experiment_Qwen_Small',
#         'params': {
#             'models.small_model': 'qwen2.5-3b-instruct',
#             'api.small_api_base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
#             'api.small_key_path': 'usage/qwen'
#         }
#     },
#     {
#         'name': 'Experiment_Qwen_Small_llama_3b_Router',
#         'params': {
#             'models.router_model': 'meta-llama/llama-3.2-3b-instruct',
#             'api.router_key_path': 'usage/openrouter1',
#             'api.router_api_base_url': 'https://openrouter.ai/api/v1',
#             'models.small_model': 'qwen2.5-3b-instruct',
#             'api.small_api_base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
#             'api.small_key_path': 'usage/qwen'
#         }
#     },
#     {
#         'name': 'Experiment_Qwen_Small_llama_1b_Router',
#         'params': {
#             'models.router_model': 'meta-llama/llama-3.2-1b-instruct',
#             'api.router_key_path': 'usage/openrouter1',
#             'api.router_api_base_url': 'https://openrouter.ai/api/v1',
#             'models.small_model': 'qwen2.5-3b-instruct',
#             'api.small_api_base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
#             'api.small_key_path': 'usage/qwen'
#         }
#     }
# ]

EXPERIMENTS_TO_RUN = [
    {
        'name': 'Experiment_AIME24_large_model1',
        'params': {
            'dataset.path': 'dataset/TestData/AIME24.json'
        }
    },
    {
        'name': 'Experiment_AIME24_large_model2',
        'params': {
            'dataset.path': 'dataset/TestData/AIME24.json'
        }
    }
]

# --- 2. 脚本设置 ---
CONFIG_FILE = 'config.yaml'
BACKUP_FILE = 'config.yaml.bak'
MAIN_SCRIPT = 'main.py'
LOG_DIR = 'logs'

def modify_config(experiment_config: dict, logger):
    """
    读取、修改并写回 config.yaml 文件。
    """
    logger.info(f"--- Modifying config for experiment: {experiment_config['name']} ---")
    try:
        # 每次都从备份的原始文件中读取，确保实验独立
        with open(BACKUP_FILE, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)

        # 应用当前实验的参数修改
        for key, value in experiment_config['params'].items():
            keys = key.split('.')
            d = config_data
            for k in keys[:-1]:
                d = d[k]
            d[keys[-1]] = value
            logger.info(f"Set '{key}' to '{value}'")

        # 将修改后的配置写入文件
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, allow_unicode=True, sort_keys=False)
        
        logger.info(f"'{CONFIG_FILE}' updated successfully for {experiment_config['name']}.")
        return True
    except Exception as e:
        logger.error(f"Error modifying config file '{CONFIG_FILE}': {e}", exc_info=True)
        return False

def run_main_script(logger):
    """
    执行 main.py 脚本并捕获其输出，然后记录结果。
    """
    logger.info(f"--- Running {MAIN_SCRIPT} ... ---")
    try:
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        # 使用 subprocess.run 来执行脚本
        result = subprocess.run(
            ['python', MAIN_SCRIPT],
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            env=env
        )
        logger.info(f"--- {MAIN_SCRIPT} executed successfully ---")
        
        # 记录标准输出
        logger.info(f"--- STDOUT for {MAIN_SCRIPT} ---")
        for line in result.stdout.strip().splitlines():
            logger.info(line)
        logger.info(f"--- End of STDOUT ---")
        return True
        
    except subprocess.CalledProcessError as e:
        # 如果脚本执行失败，记录错误信息
        logger.error(f"--- ERROR: {MAIN_SCRIPT} failed with return code {e.returncode} ---")
        logger.error(f"--- STDOUT from failed run ---")
        for line in e.stdout.strip().splitlines():
            logger.error(line)
        logger.error(f"--- STDERR from failed run ---")
        for line in e.stderr.strip().splitlines():
            logger.error(line)
        logger.error(f"--- End of error logs ---")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred while running the script: {e}", exc_info=True)
        return False

def main():
    """
    自动化测试的主函数。
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_filename = f"experiment_run_{timestamp}.log"
    setup_logger(LOG_DIR, log_filename)
    logger = get_logger()
    
    logger.info("="*80)
    logger.info("Starting new experiment run...")
    logger.info(f"Log file for this run: {os.path.join(LOG_DIR, log_filename)}")
    logger.info("="*80)

    if not os.path.exists(CONFIG_FILE):
        logger.error(f"Configuration file '{CONFIG_FILE}' not found. Aborting.")
        return

    # 备份原始配置文件
    logger.info(f"Backing up original config to '{BACKUP_FILE}'...")
    shutil.copy(CONFIG_FILE, BACKUP_FILE)
    logger.info("Backup complete.")

    try:
        print("Starting experiments...")
        experiment_iterator = tqdm(EXPERIMENTS_TO_RUN, desc="Overall Progress", unit="experiment")
        
        for experiment in experiment_iterator:
            experiment_iterator.set_postfix_str(f"Testing: {experiment['name']}")
            
            logger.info("\n" + "="*80 + "\n")
            logger.info(f"Starting test for: {experiment['name']}")
            
            # 1. 修改配置文件（基于原始备份）
            if not modify_config(experiment, logger):
                logger.error("Skipping execution due to config modification failure.")
                continue
            
            time.sleep(1) # 短暂等待，确保文件写入完成

            # 2. 运行主脚本
            run_main_script(logger)
            
            logger.info(f"Finished test for: {experiment['name']}")

    finally:
        # 3. 恢复原始配置文件
        logger.info("\n" + "="*80 + "\n")
        logger.info(f"All tests finished. Restoring original config from '{BACKUP_FILE}'...")
        if os.path.exists(BACKUP_FILE):
            shutil.move(BACKUP_FILE, CONFIG_FILE)
            logger.info("Original configuration restored.")
        logger.info("="*80)

if __name__ == '__main__':
    main()

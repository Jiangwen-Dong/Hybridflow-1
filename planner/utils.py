import os
import time
import pathlib

def build_report_path(base_dir="data_reports", is_dataset=True, dataset_name="", config=None, timestamp=None):
    """
    构建层次化的报告路径
    """
    if timestamp is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
    def clean_name(name):
        if name is None: return "unknown"
        if "/" in name: name = name.split("/")[-1]
        return ''.join(c for c in name if c.isalnum() or c in '_-.')
    
    router_name = "local_router" if config and config.use_local_router else clean_name(config.router_model if config else None)
    large_model = clean_name(config.large_model if config else None)
    small_model = clean_name(config.small_model if config else None)
    
    path_parts = [base_dir]
    if is_dataset:
        dataset_base_name = pathlib.Path(dataset_name).stem if dataset_name else "unknown_dataset"
        path_parts.extend(["dataset", dataset_base_name, router_name, large_model, small_model, timestamp])
    else:
        path_parts.extend(["single", router_name, large_model, small_model, timestamp])
    
    full_path = os.path.join(*path_parts)
    os.makedirs(full_path, exist_ok=True)
    return full_path
import networkx as nx
from collections import defaultdict, deque

def create_dag_from_tasks(tasks):
    """
    从任务字典创建有向无环图(DAG)
    
    Args:
        tasks: 任务字典，包含任务ID和依赖关系
    
    Returns:
        networkx DiGraph对象
    """
    G = nx.DiGraph()
    
    # 收集所有节点并构建边
    for step_id, attrs in tasks.items():
        if not G.has_node(step_id):
            G.add_node(step_id)
        
        rely_str = attrs.get('Rely', '')
        if rely_str:
            for dep in rely_str.split(','):
                if dep.strip():  # 确保依赖项不为空
                    G.add_edge(dep, step_id)  # 依赖指向节点
    
    # 执行传递约简，去除冗余边
    try:
        reduced_G = nx.transitive_reduction(G)
        return reduced_G
    except:
        # 如果图中有环，传递约简可能会失败
        return G

def calculate_depth_from_tasks(tasks):
    """
    计算每个步骤的深度和获得最终结果所需的总步数
    
    Args:
        tasks: 任务字典
    
    Returns:
        depths: 每个步骤的深度字典
        total_steps: 到达最后一步所需的总步数
    """
    # 创建DAG并进行传递约简
    G = create_dag_from_tasks(tasks)
    
    # 提取边列表
    edges = list(G.edges())
    
    # 初始化入度字典和邻接表
    in_degree = defaultdict(int)
    adjacency_list = defaultdict(list)
    
    nodes = set(G.nodes())
    
    # 填充邻接表和计算入度
    for u, v in edges:
        adjacency_list[u].append(v)
        in_degree[v] += 1
    
    # 找出所有入度为0的节点（根节点）
    queue = deque([node for node in nodes if in_degree[node] == 0])
    depths = {node: 0 for node in queue}  # 根节点深度为0
    
    # 广度优先搜索计算深度
    while queue:
        node = queue.popleft()
        current_depth = depths[node]
        
        for neighbor in adjacency_list[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
                # 更新为最大深度
                depths[neighbor] = current_depth + 1
    
    # 为所有没有被赋值深度的节点设置默认值
    for node in nodes:
        if node not in depths:
            # 可能存在的环或者孤立节点
            depths[node] = 0
    
    # 找出最大深度，即总步数（+1 因为最小深度为0）
    total_steps = max(depths.values()) + 1 if depths else 0
    
    return depths, total_steps

def calculate_task_metrics(tasks):
    """
    计算任务的各种指标
    
    Args:
        tasks: 任务字典
    
    Returns:
        dict: 包含各种指标的字典
    """
    # 1. 总步骤数
    total_tasks_num = len(tasks)
    
    # 2. 计算DAG深度和压缩比
    depths, max_depth = calculate_depth_from_tasks(tasks)
    compression_ratio = max_depth / total_tasks_num if total_tasks_num > 0 else 0
    
    # 3. 计算平均token数量
    total_tokens = 0
    token_count = 0
    
    for step_id, attrs in tasks.items():
        if 'Token' in attrs:
            try:
                tokens = int(attrs['Token'])
                total_tokens += tokens
                token_count += 1
            except (ValueError, TypeError):
                # 如果无法转换为整数，则跳过
                pass
    
    avg_tokens = total_tokens / token_count if token_count > 0 else 0
    
    return {
        "total_tasks_num": total_tasks_num,
        "max_depth": max_depth,
        "compression_ratio": compression_ratio,
        "avg_task_plan_tokens": avg_tokens,
    }

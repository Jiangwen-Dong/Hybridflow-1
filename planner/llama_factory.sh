source /etc/network_turbo

git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
pip install swanlab

pip install --no-deps -e .

llamafactory-cli version
llamafactory-cli webui
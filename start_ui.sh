#!/bin/bash
# Deblur-NeRF UI - WSL 启动脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 激活虚拟环境
if [ ! -d "venv" ]; then
    echo "⚙️ 初始化虚拟环境..."
    python3 -m venv venv
    source venv/bin/activate
    python3 -m pip install --upgrade pip -q
    
    echo "📦 安装所有依赖包..."
    python3 -m pip install -q -r requirements.txt
    
    echo "✅ 虚拟环境初始化完成"
else
    source venv/bin/activate
    # 定期更新依赖 (可选)
    python3 -m pip install -q -r requirements.txt 2>/dev/null || true
fi

echo ""
echo "🎬 Deblur-NeRF UI 启动器"
echo "====================================="
echo ""
echo "✓ Python 版本: $(python3 --version 2>&1 | cut -d' ' -f2)"
echo "✓ Streamlit: $(python3 -c 'import streamlit; print(streamlit.__version__)' 2>/dev/null)"
echo ""

# 获取WSL的IP地址
WSL_IP=$(hostname -I | awk '{print $1}')

echo "🌐 网络配置:"
echo "   WSL IP:      $WSL_IP"
echo "   WSL 端口:    8501"
echo ""
echo "📍 访问地址:"
echo "   从 Windows:  http://$WSL_IP:8501"
echo "   从 WSL:      http://localhost:8501"
echo ""
echo "💡 如果仍无法连接:"
echo "   1. 检查Windows防火墙是否阻止了8501端口"
echo "   2. 尝试: netsh advfirewall firewall add rule name='Streamlit' dir=in action=allow protocol=tcp localport=8501"
echo ""
echo "🚀 正在启动 Deblur-NeRF UI..."
echo "📍 按 CTRL+C 停止服务"
echo ""

streamlit run ui.py --server.address 0.0.0.0 --server.port 8501

#!/bin/bash
# auto_extract_smartspace.sh
# 智能空间管理提取脚本 - 自动检查、清理、循环提取

set -e

cd /home/ncl/workspace/Deblur-NeRF

echo "🚀 Deblur-NeRF 智能提取脚本 (空间管理版)"
echo "=========================================="
echo ""

# 函数：获取可用空间 (MB)
get_free_space() {
    df /home/ncl/workspace/ | awk 'NR==2 {printf "%.0f", $4/1024}'
}

# 函数：获取目录大小 (MB)
get_dir_size() {
    du -sm "$1" 2>/dev/null | awk '{print $1}'
}

# 函数：格式化字节为MB/GB
format_size() {
    if [ $1 -gt 1024 ]; then
        echo "$((($1 + 512) / 1024))GB"
    else
        echo "${1}MB"
    fi
}

# 检查磁盘空间
echo "💾 磁盘空间检查"
echo "==============="
free_mb=$(get_free_space)
echo "当前可用空间: $(format_size $free_mb)"
echo ""

# 显示各数据集大小
echo "📦 各数据集信息"
echo "==============="
declare -A datasets
datasets[pretrainweights]=235
datasets[real_camera_motion_blur]=550
datasets[real_defocus_blur]=691
datasets[real_object_motion_blur]=67
datasets[synthetic_camera_motion_blur]=62
datasets[synthetic_defocus_blur]=74
datasets[synthetic_gt]=89
datasets[blend_files]=89

for name in pretrainweights real_camera_motion_blur real_defocus_blur real_object_motion_blur synthetic_camera_motion_blur synthetic_defocus_blur synthetic_gt blend_files; do
    size=${datasets[$name]}
    zip_file=$(find photo -name "*${name}*.zip" 2>/dev/null | head -1)
    
    if [ -f "$zip_file" ]; then
        if [ $free_mb -ge $size ]; then
            status="✅ 可提取"
        else
            status="❌ 空间不足"
        fi
        echo "  $name: $(format_size $size) $status"
    fi
done
echo ""

# 空间不足处理
if [ $free_mb -lt 235 ]; then
    echo "❌ 严重错误: 可用空间不足235MB!"
    echo "请至少留出 500MB 空间"
    exit 1
fi

if [ $free_mb -lt 785 ]; then
    echo "⚠️  空间警告: 只能提取小数据集"
    echo ""
    echo "可行方案:"
    echo "  1) 只提取预训练权重 (235MB)"
    echo "  2) 提取后删除ZIP以释放空间"
    echo ""
fi

echo "请选择提取方案:"
echo "1) 仅预训练权重 (235MB)"
echo "2) 预训练 + 相机模糊 (785MB) [需要至少 900MB]"
echo "3) 智能循环提取 [逐个提取+删除ZIP] ⭐"
echo "4) 查看磁盘清理建议"
echo ""

read -p "请选择 (1/2/3/4) [默认: 3]: " choice
choice=${choice:-3}

case $choice in
    1)
        echo ""
        echo "📥 提取方案: 仅预训练权重"
        echo "需要空间: 235MB (可用: $(format_size $free_mb))"
        echo ""
        
        if [ $free_mb -lt 235 ]; then
            echo "❌ 空间不足!"
            exit 1
        fi
        
        mkdir -p pretrained_weights
        
        echo "📦 提取中..."
        unzip -q photo/pretrainweights-*.zip -d pretrained_weights/
        echo "✅ 完成!"
        ;;
    2)
        echo ""
        echo "📥 提取方案: 预训练 + 相机模糊"
        echo "需要空间: 785MB (可用: $(format_size $free_mb))"
        echo ""
        
        if [ $free_mb -lt 785 ]; then
            echo "❌ 空间不足!"
            echo "建议选择方案3 (智能循环提取)"
            exit 1
        fi
        
        mkdir -p data pretrained_weights
        
        echo "📦 提取预训练权重..."
        unzip -q photo/pretrainweights-*.zip -d pretrained_weights/
        echo "✅ pretrainweights 完成"
        
        echo "📦 提取相机模糊数据..."
        unzip -q photo/real_camera_motion_blur-*.zip -d data/
        echo "✅ real_camera_motion_blur 完成"
        ;;
    3)
        echo ""
        echo "📥 智能循环提取方案 ⭐"
        echo "=========================================="
        echo "说明: 逐个提取 → 删除ZIP → 释放空间 → 继续"
        echo ""
        
        mkdir -p data pretrained_weights
        
        # 提取顺序 (按优先级)
        datasets_to_extract=(
            "pretrainweights:235"
            "real_camera_motion_blur:550"
            "real_defocus_blur:691"
            "real_object_motion_blur:67"
            "synthetic_camera_motion_blur:62"
            "synthetic_defocus_blur:74"
            "synthetic_gt:89"
        )
        
        for item in "${datasets_to_extract[@]}"; do
            IFS=':' read -r dataset_name dataset_size <<< "$item"
            
            # 检查ZIP是否存在
            zip_file=$(find photo -name "*${dataset_name}*.zip" 2>/dev/null | head -1)
            if [ ! -f "$zip_file" ]; then
                echo "⏭️  跳过 $dataset_name (ZIP不存在)"
                continue
            fi
            
            # 检查是否已提取 (解压后的文件夹)
            if [ -d "data/$dataset_name" ] || [ -d "pretrained_weights/$dataset_name" ]; then
                echo "⏭️  跳过 $dataset_name (已提取)"
                continue
            fi
            
            free_mb=$(get_free_space)
            echo ""
            echo "📦 处理: $dataset_name (需要 $(format_size $dataset_size), 可用 $(format_size $free_mb))"
            
            if [ $free_mb -lt $dataset_size ]; then
                echo "⚠️  空间不足 ($(format_size $dataset_size) > $(format_size $free_mb))"
                
                read -p "是否删除ZIP以释放空间后重试? (y/n) [默认: y]: " del_choice
                del_choice=${del_choice:-y}
                
                if [[ "$del_choice" == "y" || "$del_choice" == "Y" ]]; then
                    # 删除最大的已提取数据的ZIP来释放空间 (保留1个作为备份)
                    largest_zip=$(ls -hS photo/*.zip 2>/dev/null | head -1 | awk '{print $NF}')
                    if [ -f "$largest_zip" ]; then
                        largest_name=$(basename "$largest_zip" | sed 's/-[0-9]*Z.*//')
                        
                        # 检查该数据是否已提取
                        if [ -d "data/$largest_name" ] || [ -d "pretrained_weights/$largest_name" ]; then
                            echo "🗑️  删除: $(basename $largest_zip)"
                            rm "$largest_zip"
                            freed=$(($(du -sm photo | awk '{print $1}' 2>/dev/null || echo 0)))
                            echo "✅ 释放space，重试中..."
                            continue
                        fi
                    fi
                    echo "❌ 无法释放足够空间"
                else
                    echo "⏭️  跳过此数据集"
                    continue
                fi
            fi
            
            # 提取
            if [[ "$dataset_name" == "pretrainweights" ]]; then
                unzip -q "$zip_file" -d pretrained_weights/
            else
                unzip -q "$zip_file" -d data/
            fi
            
            echo "✅ $dataset_name 提取完成"
            
            # 询问是否删除ZIP以释放空间
            free_mb_now=$(get_free_space)
            if [ $free_mb_now -lt 300 ]; then
                read -p "磁盘空间不足 ($(format_size $free_mb_now))，删除ZIP文件以释放空间? (y/n) [默认: y]: " delete_zip
                delete_zip=${delete_zip:-y}
                
                if [[ "$delete_zip" == "y" || "$delete_zip" == "Y" ]]; then
                    echo "🗑️  删除 $(basename $zip_file)..."
                    rm "$zip_file"
                    freed_size=$((dataset_size))
                    echo "✅ 释放 $(format_size $freed_size) 空间"
                    echo "   新可用空间: $(format_size $(get_free_space))"
                fi
            fi
        done
        ;;
    4)
        echo ""
        echo "🧹 磁盘清理建议"
        echo "================"
        echo ""
        echo "目前占用最大的目录:"
        du -sh /home/ncl/workspace/Deblur-NeRF/* 2>/dev/null | sort -rh | head -5
        echo ""
        echo "💡 清理方案:"
        echo ""
        echo "方案A: 删除__pycache__ (76KB)"
        echo "  rm -rf /home/ncl/workspace/Deblur-NeRF/__pycache__"
        echo ""
        echo "方案B: 只保持必要的预训练权重"
        echo "  1. 将其他ZIP先删除"
        echo "  2. 保留最需要的2-3个ZIP"
        echo "  3. 需要时再从云端重新下载"
        echo ""
        echo "方案C: 检查系统级别的磁盘清理"
        echo "  统计: $(du -sh /home/ncl/workspace/Deblur-NeRF 2>/dev/null | awk '{print $1}') 已用"
        echo "        $(free -h | grep Mem | awk '{print $7}') 可用内存"
        exit 0
        ;;
    *)
        echo "❌ 无效选择"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "✅ 提取完成！"
echo "=========================================="
echo ""

# 最终统计
echo "📊 最终统计:"
free_mb_final=$(get_free_space)
echo "  剩余可用空间: $(format_size $free_mb_final)"

if [ -d "data" ]; then
    data_size=$(get_dir_size data)
    echo "  data/ 大小: $(format_size $data_size)"
fi

if [ -d "pretrained_weights" ]; then
    weights_size=$(get_dir_size pretrained_weights)
    echo "  pretrained_weights/ 大小: $(format_size $weights_size)"
fi

echo ""
echo "👉 后续步骤:"
echo "  streamlit run ui.py"
echo ""

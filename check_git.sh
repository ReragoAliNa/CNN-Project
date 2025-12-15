#!/bin/bash

# 定义颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=========================================${NC}"
echo -e "${YELLOW}      Git Push 问题自动诊断工具      ${NC}"
echo -e "${YELLOW}=========================================${NC}"

# 1. 检查是否在 Git 仓库中
if [ ! -d ".git" ]; then
    echo -e "${RED}[x] 错误: 当前目录不是 Git 仓库。请在项目根目录下运行此脚本。${NC}"
    exit 1
fi

# 2. 获取远程仓库地址
REMOTE_URL=$(git remote get-url origin 2>/dev/null)
if [ -z "$REMOTE_URL" ]; then
    echo -e "${RED}[x] 错误: 未找到远程仓库 'origin'。${NC}"
    exit 1
else
    echo -e "${GREEN}[√] 远程仓库地址: $REMOTE_URL${NC}"
fi

# 3. 检查网络连通性 (Ping)
echo -e "\n${YELLOW}正在检查 GitHub 网络连通性 (Ping)...${NC}"
if ping -c 3 github.com &> /dev/null; then
    echo -e "${GREEN}[√] 可以连接到 github.com (Ping 成功)${NC}"
else
    echo -e "${RED}[x] 警告: 无法 Ping 通 github.com。可能存在网络阻断或防火墙问题。${NC}"
fi

# 4. 检查 SSH 连接 (如果是 SSH 协议)
if [[ "$REMOTE_URL" == git@* ]]; then
    echo -e "\n${YELLOW}正在测试 SSH 连接 (ssh -T git@github.com)...${NC}"
    ssh -T -o ConnectTimeout=10 git@github.com 2>&1 | grep "successfully authenticated" &> /dev/null
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}[√] SSH 认证成功！连接正常。${NC}"
    else
        echo -e "${RED}[x] SSH 连接失败！请检查 SSH Key 或网络代理。${NC}"
        echo -e "    尝试运行命令: ssh -v git@github.com 查看详细报错"
    fi
else
    echo -e "\n${YELLOW}检测到使用 HTTPS 协议，跳过 SSH 检查。${NC}"
fi

# 5. 检查 Git 代理设置
echo -e "\n${YELLOW}检查 Git 代理设置...${NC}"
HTTP_PROXY=$(git config --global --get http.proxy)
HTTPS_PROXY=$(git config --global --get https.proxy)

if [ -z "$HTTP_PROXY" ] && [ -z "$HTTPS_PROXY" ]; then
    echo -e "${GREEN}[i] 未检测到 Git 全局代理 (通常是好的，除非你需要翻墙)。${NC}"
else
    echo -e "${YELLOW}[!] 检测到代理设置:${NC}"
    echo "    http.proxy: $HTTP_PROXY"
    echo "    https.proxy: $HTTPS_PROXY"
    echo -e "${YELLOW}    如果网络不通，请尝试取消代理: git config --global --unset http.proxy${NC}"
fi

# 6. 检查是否有大文件 (这是导致 send-pack error 的常见原因)
echo -e "\n${YELLOW}正在扫描是否有超过 50MB 的大文件...${NC}"
# 获取所有打包的对象，按大小排序，取最大的5个
# 注意：这需要 git verify-pack，通常 git 自带
if command -v git-verify-pack &> /dev/null; then
    big_files=$(git verify-pack -v .git/objects/pack/*.idx 2>/dev/null | sort -k 3 -n | tail -5)
    
    if [ -z "$big_files" ]; then
         echo -e "${GREEN}[√] 未检测到异常大的对象。${NC}"
    else
         echo -e "${YELLOW}[!] 发现仓库中较大的对象 (最后 5 个):${NC}"
         echo "$big_files" | awk '{printf "    Size: %.2f MB | SHA: %s\n", $3/1024/1024, $1}'
         echo -e "${YELLOW}    注意: 如果有单文件超过 100MB (GitHub限制) 或 50MB (警告)，推送可能会中断。${NC}"
    fi
else
    echo -e "${YELLOW}[i] 无法运行 git verify-pack，跳过大文件检测。${NC}"
fi

# 7. 给出建议
echo -e "\n${YELLOW}=========================================${NC}"
echo -e "${YELLOW}            诊断结束 & 建议              ${NC}"
echo -e "${YELLOW}=========================================${NC}"
echo "1. 如果 SSH 失败但 Ping 成功: 你的网络可能阻断了 SSH (端口 22)。"
echo "   -> 尝试配置 SSH config 使用 443 端口。"
echo "2. 如果有大文件 (>100MB):"
echo "   -> 你必须撤销该提交，或者使用 Git LFS。"
echo "3. 如果一切看起来都正常，但依然报错:"
echo "   -> 尝试增加 Git 缓存并配置 SSH 保活 (执行以下命令):"
echo -e "${GREEN}   git config --global http.postBuffer 524288000${NC}"
echo -e "${GREEN}   (如果不使用 SSH 可忽略) 在 ~/.ssh/config 中加入 ServerAliveInterval 60${NC}"
echo ""
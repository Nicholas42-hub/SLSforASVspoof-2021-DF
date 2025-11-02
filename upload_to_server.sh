#!/bin/bash
# 上传更新的文件到远程服务器
# Upload updated files to remote server

# 替换为您的服务器地址
# Replace with your server address
SERVER="root@your-server-address"
REMOTE_DIR="/root/autodl-tmp/SLSforASVspoof-2021-DF"

echo "📤 Uploading files to server..."
echo "Server: $SERVER"
echo "Remote directory: $REMOTE_DIR"
echo ""

# 上传3个核心文件
echo "1️⃣  Uploading evaluate_with_attention_viz.py..."
scp evaluate_with_attention_viz.py $SERVER:$REMOTE_DIR/

echo "2️⃣  Uploading visualize_attention_evaluation.py..."
scp visualize_attention_evaluation.py $SERVER:$REMOTE_DIR/

echo "3️⃣  Uploading run_incorrect_viz.sh..."
scp run_incorrect_viz.sh $SERVER:$REMOTE_DIR/

echo ""
echo "✅ Upload complete!"
echo ""
echo "Next steps on server:"
echo "  cd $REMOTE_DIR"
echo "  chmod +x run_incorrect_viz.sh"
echo "  ./run_incorrect_viz.sh"

#!/bin/bash

# 检查是否传递了消息内容作为参数
if [ $# -lt 1 ]; then
  echo "Usage: $0 '<message content>'"
  exit 1
fi

# 设置请求的URL
url="https://wxpusher.zjiecode.com/api/send/message"

# 获取传递的消息内容
content=$1

# 设置请求的JSON数据
json_data=$(cat <<EOF
{
  "appToken":"xxx",
  "content":"$content",
  "summary":"消息摘要",
  "contentType":2,
  "topicIds":[123],
  "uids":["UID_zyOFdJlAu39zJzqGqBN9ItLGl42v"],
  "url":"https://wxpusher.zjiecode.com",
  "verifyPay":false,
  "verifyPayType":0
}
EOF
)

# 发送POST请求
curl -X POST "$url" \
     -H "Content-Type:application/json" \
     -d "$json_data"
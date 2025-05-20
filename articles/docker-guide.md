# Docker 入门指南

## 基本命令
```bash
# 运行容器
docker run hello-world

# 查看运行中的容器
docker ps

# 构建镜像
docker build -t myapp .
```

## Dockerfile 示例
```dockerfile
FROM node:16
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
EXPOSE 3000
CMD ["npm", "start"]
``` 
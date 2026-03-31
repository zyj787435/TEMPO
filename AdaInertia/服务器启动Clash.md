# 🚀 VS Code SSH + Claude 代理启动速查表

**适用场景**：服务器重启、SSH 断开重连、或 Claude 插件突然报错时。

## 1. 连接服务器
在桌面上启动`EasierConnect`，启动学校VPN，然后在[AutoDL](https://private.autodl.com/console/instance)开机并复制密码。

使用VSCode侧边栏的远程资源管理器，或按`F1`连接到`nju-server`

---

## 2. 启动代理

利用`tmux`新建一个会话来运行 Clash(Mihomo)。

1. **新建 tmux 会话**：
```bash
tmux new -s clash
```
> 若提示会话重复，利用`tmux kill-session -t clash`结束


2. **运行 Mihomo** (在进入的新窗口中)：
```bash
sudo /etc/mihomo/mihomo -d /etc/mihomo -f /etc/mihomo/config.yaml
```

3. 当前终端已经被Mihomo占用，**打开一个新的终端标签页**查看代理是否挂上
```bash
curl -x http://127.0.0.1:7890 ipinfo.io/country
```
- 如果返回`HK`或者其他代码，说明代理已成功启动，接下来需要切换到**美国节点**。

## 3. 切换节点
1. 打开端口标签页

![alt text](<截屏2026-02-06 14.43.45.png>)

2. 添加端口`9090`

![alt text](<截屏2026-02-06 14.46.52.png>)

3. 点击访问➡️<a href="https://metacubex.github.io/metacubexd" style="color: #FFACC4; text-decoration: underline;">Clash 控制面板</a>

4. 切换到图示节点

![alt text](<截屏2026-02-06 15.00.31.png>)

5. 返回终端，再次运行 `curl` 命令确认已经切换到美国节点。
```bash
curl -x http://127.0.0.1:7890 ipinfo.io/country
```
- 如果返回`US`，说明节点切换成功，可以使用Claude插件了

---

## 常见问题
### VSCode代理配置
如果第 1 步检测通过（curl 能通），但 Claude 依然报错，请检查 VS Code 远程设置。

1. 按 `F1` -> 输入 `Remote Settings (JSON)`。
2. 确保包含以下内容：
```json
{
    "http.proxy": "http://127.0.0.1:7890",
    "http.proxySupport": "on",
    "http.proxyStrictSSL": false
}
```

### 更新订阅
如果是机场链接过期或更换，在本机复制`yaml`文件内容并覆盖旧配置，**然后重启 Mihomo**。

### 完全停止代理
```bash
sudo pkill mihomo
```

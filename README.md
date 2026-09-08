# AI-RAN NPU Kernels

AI-RAN 通信算法的 NPU 算子与收发链实现。

主要目录：`chain`、`channel_est`、`equalization`、`fec`、`mapping`、`mimo`、`ofdm` 和 `sync`。

## 作者与团队

- 作者：张世龙，南京大学博士生
- 单位：南京大学、南苏智网实验室（NINE）
- 团队：杨鲲团队
- 导师：向路平、陈杰男（电子科技大学通信抗干扰全国重点实验室）

## 项目内容

本项目面向 AI-RAN 通信基带处理，完成了昇腾 NPU 上的模块化算子和端到端收发链实现，主要包括：

- 同步、频偏估计与补偿、OFDM 调制与解调；
- 信道估计、均衡、MIMO 检测与预编码；
- 层映射、资源映射、QAM 调制与解调；
- LDPC 编解码、速率匹配与解匹配、加扰与解扰；
- PUSCH SISO/MIMO TX/RX 收发链及 Rank 1–4 多天线配置；
- AscendC kernel、tiling、主机端调用、参考算法和自动化验证脚本。

项目采用模块化目录组织，既可独立构建和验证单个算子，也可运行完整通信链路。

## 使用方法

准备好昇腾 CANN 环境，并设置工具链路径：

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
source "$ASCEND_HOME_PATH/set_env.sh"
```

进入任一算子目录后执行：

```bash
bash run.sh
```

运行完整收发链：

```bash
bash chain/pusch_siso_tx_chain/run.sh
bash chain/pusch_siso_rx_chain/run.sh
bash chain/pusch_mimo_tx_chain/run.sh 4 23
bash chain/pusch_mimo_rx_chain/run.sh
```

运行前请根据对应脚本准备测试数据和权重；默认目标平台为 Ascend 310P1。

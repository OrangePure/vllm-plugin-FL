# enable_static_kernel（ACLGraph 静态算子预编译）在 vllm-plugin-FL 的适配

> 目标：对齐 vllm-ascend main 的 `enable_static_kernel`（`ascend_config.py`、
> `compilation/compiler_interface.py`、`worker/worker.py`），在 npugraph_ex/torchair
> ACLGraph 捕获阶段按 decode batch 档位把算子预编译为固定 shape 的静态 kernel，
> 消除 replay 时的动态 shape/tiling 开销。
> 实现文件：
> - `vllm_fl/dispatch/backends/vendor/ascend/compilation/compiler_interface.py`（编译选项主体）
> - `vllm_fl/dispatch/backends/vendor/ascend/patches/patch_static_kernel.py`（配置判定与退出卸载钩子）

---

## 1. 开启方式

与 vllm-ascend 相同的 `additional_config` key（插件刻意保持同名）：

```bash
vllm serve <model> \
  --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":true, "enable_static_kernel":true}}'
```

或环境变量：`VLLM_FL_ENABLE_NPUGRAPH_EX=1` + `VLLM_FL_ENABLE_STATIC_KERNEL=1`。

约束（与上游一致）：

- **必须同时开启 `enable_npugraph_ex`**，否则忽略并告警（上游是 assert，插件降级为 warning）。
- 310P 上强制忽略（npugraph_ex 不支持 310P）。
- 依赖环境中安装 `npugraph_ex`（优先）或 `torchair`；都没有则 `AscendCompiler` 编译时报错提示关闭。
- 首次启动会在图捕获阶段触发静态编译，**启动时间增加数分钟到数十分钟**。

## 2. 机制（对照 vllm-ascend main）

| 环节 | 行为 |
|---|---|
| 配置 | `patch_static_kernel.is_static_kernel_enabled()`：env 或 `additional_config` 读取，校验 npugraph_ex / 310P |
| 编译选项 | `AscendCompiler.compile()` → `npugraph_ex_compile()` → `_configure_backend()`：npugraph_ex 路径下发 `static_kernel_compile=True` + `_vllm_aclnn_static_kernel_sym_range`；torchair 路径设置 `experimental_config.aclgraph._aclnn_static_shape_kernel=True` + `_aclnn_static_shape_kernel_sym_value_range` |
| 编译范围 | `_compute_decode_cudagraph_batch_sizes()`：`cudagraph_capture_sizes` 过滤到 `[spec_tokens+1, max_num_seqs*(spec_tokens+1)]`，只对 decode 档位做静态编译 |
| 多机/多卡协调 | 开启时自动设置 `LOCAL_WORLD_SIZE = local_world_size * data_parallel_size_local`（npugraph_ex 据此建节点内 Gloo group 协调编译与 .run 包安装） |
| 缓存 key | `AscendCompiler.compute_hash` 纳入 `enable_npugraph_ex` / `enable_static_kernel` |
| 退出卸载 | 静态 kernel 安装为 `$ASCEND_HOME_PATH/opp/static_kernel` 下的 .run 包（进程外全局状态），`apply_ascend_patches()` 在 worker 初始化时调用 `patch_static_kernel.patch_static_kernel()`，开启该特性则注册 SIGTERM/SIGINT 钩子，退出时以 fcntl 文件锁保证单进程执行 `ai_core/uninstall.sh` |

与 site-packages 版 vllm-ascend（vLLM 0.13 兼容版）的区别：该版本在 `npugraph_ex_compile`
中**硬编码** `_aclnn_static_shape_kernel=True`（无开关、无 sym_range、无 LOCAL_WORLD_SIZE、
无卸载钩子）；插件按 main 版语义实现为**默认关闭、可配置、限定 decode 档位**。

## 3. 验证方式（对齐上游）

- 启动日志出现 `Starting static kernel compilation, the build directory is ...`；
- 或 Ascend Profiling 的 `op_statistic.csv` 中 `op_type`/`name` 列出现 `static_kernel` 关键字。

注意：本仓库交付环境的 Python 未安装 torchair/npugraph_ex，端到端静态编译需在装好
torchair/npugraph_ex 的环境验证。

# DataX 任务模板说明

平台提供两种基于 [阿里巴巴 DataX](https://github.com/alibaba/DataX) 的任务模板，适用于异构数据源同步与数据导入场景。

**镜像**：`ccr.ccs.tencentyun.com/cube-studio/datax:20240501`

---

## 一、datax 模板

### 实现方式

- **入口**：`start.sh`
- **底层**：直接调用阿里巴巴 DataX，通过 `-f` 指定 job 配置文件。

### 适用场景

- 需要完整使用 DataX 能力（多种 Reader/Writer、复杂映射、过滤等）
- 已有或愿意手写 DataX job JSON 的用户
- 数据源/目标为 DataX 支持但 datax-import任务模板 未封装的类型

### 参数说明

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `-f` | 文件路径 | 是 | job.json 文件地址。JSON 书写格式请参考 [DataX 官方文档](https://github.com/alibaba/DataX) |

### 使用步骤

1. 在平台工作目录或挂载卷中准备 DataX job 配置文件（如 `job.json`）。
2. 创建 **datax** 任务，在参数中填写 **文件地址** 为该 json 的路径（例如：`/mnt/{{creator}}/pipeline/example/ml/mysql-csv.json`）。
3. 提交任务后，`start.sh` 会执行：`python datax.py <你填写的文件路径>`。

### 参考链接

- [DataX GitHub](https://github.com/alibaba/DataX)

---

## 二、datax-import 模板（表单化导入，免写 JSON文件）

### 实现方式

- **入口**：`start.py`
- **目的**：通过表单参数生成 DataX job 配置，无需手写复杂 JSON，将数据库表导出为 CSV。

### 适用场景

- 从 **MySQL / PostgreSQL / ClickHouse** 等数据库表导出到 **CSV 文件**
- 希望用「数据库类型、连接信息、表名、列名、保存路径」等表单填空即可运行，无需接触 JSON

### 参数说明

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `--db_type` | 选择 | 是 | 数据库类型：`mysql`、`postgresql`、`clickhouse` |
| `--host` | 字符串 | 是 | 数据库地址（含端口，如 `mysql-service.infra:3306`） |
| `--username` | 字符串 | 是 | 数据库用户名 |
| `--password` | 字符串 | 是 | 数据库密码 |
| `--database` | 字符串 | 是 | 库名 |
| `--table` | 字符串 | 是 | 表名 |
| `--columns` | 字符串 | 是 | 要导出的列名，逗号分隔（如 `id,name,age`） |
| `--save_path` | 文件路径 | 是 | 导出 CSV 的保存地址（平台工作目录或挂载路径） |
| `--query_sql` | 字符串 | 否 | 自定义查询 SQL（可选，脚本支持该参数） |

### 使用步骤

1. 在任务模板中选择 **datax-import**。
2. 在表单中填写：数据库类型、地址、用户名、密码、库名、表名、列名、保存路径（及可选查询 SQL）。
3. 提交任务后，`start.py` 会根据 `to-csv.json` 模板替换占位符，生成临时 DataX 配置并执行，将表数据导出为指定路径的 CSV。

### 与 datax 模板的关系

- 使用同一 DataX 镜像与 `datax.py`，仅入口不同：datax 用 `start.sh` + 自带 job.json；datax-import 用 `start.py` + 内置 `to-csv.json` 模板生成 job 再调用 DataX。

---

## 三、如何选择

| 需求 | 推荐模板 |
|------|----------|
| 任意 DataX 支持的读写组合、复杂配置 | **datax**（手写 job.json，用 `start.sh`） |
| 从 MySQL/PostgreSQL/ClickHouse 表导出为 CSV，快速配置 | **datax-import**（表单填写，用 `start.py`） |

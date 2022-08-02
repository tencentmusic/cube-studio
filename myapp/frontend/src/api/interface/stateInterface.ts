// export const TTaskState: Record<string, string> = {
//     "-1": "ignore",//忽略任何条件，直接执行
//     "0": "waiting",//0 是等待父任务
//     "1": "running",//正在执行
//     "2": "success",//成功
//     "3": "failed",//失败
//     "4": "sentenced",//准备kill,等待终止
//     "5": "killing",//正在kill   
//     "6": "killed",//已经kill
//     "7": "hanged",//kill_failed 夯住，kill失败
//     "8": "terminated",//永久终止
//     "9": "dependence",//finish 依赖校验结束，等待下发
//     "10": "issuing",//
//     "11": "runer",//stop runner停止
//     "12": "wait",//resource 等待资源
//     "13": "future",//value4 未应用
//     "14": "future",//value5 未应用
//     "15": "future",//value6 未应用
//     "16": "internal",//sentenced 与4对应用，但只应用于系统内部的一些操作，先不用考虑
//     "17": "internal",//killing 与5对应用，但只应用于系统内部的一些操作，先不用考虑
// }

export const TInstanceState: Record<string, string> = {
    "0": "等待父任务",//0 是等待父任务
    "1": "正在执行",//正在执行
    "2": "成功",//成功
    "3": "失败",//失败
    "5": "正在kill",//正在kill
    "8": "永久终止",//永久终止
    "9": "依赖校验结束，等待下发",//finish 依赖校验结束，等待下发
    "11": "runer",//stop runner停止
    "12": "等待资源",//resource 等待资源
}

export const TTaskType: Record<string, string> = {
    2: '数据库表同步',
    12: 'Hadoop同步至Linux',
    25: 'TL-HDFS接入Glacier',
    61: 'TDW出库Hbase',
    66: 'TDW出库到PG',
    68: 'TDW出库MySql',
    75: 'HDFS入库至TDW',
    76: 'TDW出库至HDFS',
    77: 'TDW出库OZONE',
    84: '校验对账文件',
    102: 'SQL计算',
    104: 'HDFS同步至HDFS',
    105: 'TDW_TO_HDFS_POSTGRE',
    106: '调用Shell脚本',
    107: 'TDBANK_HDFS入库至TDW',
    108: '校验对账文件tdbank',
    111: 'pg存储过程',
    116: 'TDW出库ClickHouse',
    118: 'DB导出至HDFS',
    119: 'HDFS入库至TDW_Gaia',
    120: 'caffe模型训练',
    121: 'PythonSQL脚本',
    125: '数据库写入TDBank',
    126: 'TDBank数据迁移',
    128: 'SparkScala计算',
    129: 'PySpark计算',
    130: '虫洞任务',
    132: 'SuperSQL',
    134: 'TDBANK入HIVE',
    135: 'Hbase导入TDW',
    136: 'Flink入库TDW',
    203: 'TDW出库ClickHouse(通用)',
    208: 'bulkload出库clickhouse'
}

export const TTaskState: Record<string, string> = {
    'C': '草稿', 'F': '冻结', 'Y': '正常'
}

export const TTaskCircle: Record<string, string> = {
    'M': '月',
    'W': '周',
    'D': '天',
    'H': '小时',
    'I': '分钟',
    'R': '非周期',
}
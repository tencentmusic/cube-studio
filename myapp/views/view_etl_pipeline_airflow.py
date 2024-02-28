import pysnooper
from flask_babel import gettext as __
from flask_babel import lazy_gettext as _
from myapp import app, appbuilder, db

conf = app.config


# todo： airflow的运行，删除，日志查询
class AIRFLOW_ETL_PIPELINE():

    def __init__(self, pipeline, host='http://airflow.oa.com'):
        self.pipeline = pipeline
        self.host = host

    # todo: 想要支持的模板列表
    all_template = {
        "message": "success",
        # 任务元数据配置参数
        "task_metadata_ui_config": {
            "metadata": {
                "label": {
                    "type": "str",
                    "label": __("中文名称"),
                    "require": 1,
                    "choice": [],
                    "range": "",
                    "default": "",
                    "placeholder": "",
                    "describe": __("任务中文别名"),
                    "editable": 1,
                    "addable": 1
                }
            }
        },
        # 公共配置参数，在每个任务的参数中都有
        "templte_common_ui_config": {
            __("任务元数据"): {
                "crontab": {
                    "type": "str",
                    "item_type": "str",
                    "label": __("调度周期"),
                    "require": 0,
                    "choice": [],
                    "range": "",
                    "default": "1 1 * * *",
                    "placeholder": "",
                    "describe": __("周期任务的时间设定 * * * * * 一次性任务可不填写 <br>表示为 minute hour day month week"),
                    "editable": 1,
                    "addable": 0  # 1 为仅在添加时可修改
                },
                "selfDepend": {
                    "type": "str",
                    "item_type": "str",
                    "label": __("自依赖判断"),
                    "require": 1,
                    "choice": [__("自依赖"), __('单实例运行'), __('多实例运行')],
                    "range": "",
                    "default": __("单实例运行"),
                    "placeholder": "",
                    "describe": __("一个任务的多次调度实例之间是否要进行前后依赖"),
                    "editable": 1,
                    "addable": 0  # 1 为仅在添加时可修改
                },
                "ResourceGroup": {
                    "type": "str",
                    "item_type": "str",
                    "label": __("队列"),
                    "require": 1,
                    "choice": [item for item in ['default', 'queue1', 'queue2']],
                    "range": "",
                    "default": 'default',
                    "placeholder": "",
                    "describe": __("队列"),
                    "editable": 1,
                    "addable": 0
                }
            },
            __("监控配置"): {
                "alert_user": {
                    "type": "str",
                    "item_type": "str",
                    "label": __("报警用户"),
                    "require": 0,
                    "choice": [],
                    "range": "",
                    "default": "admin,",
                    "placeholder": "",
                    "describe": __("报警用户，逗号分隔"),
                    "editable": 1,
                    "addable": 0  # 1 为仅在添加时可修改
                },
                "timeout": {
                    "type": "str",
                    "item_type": "str",
                    "label": __("超时中断"),
                    "require": 1,
                    "choice": [],
                    "range": "",
                    "default": "0",
                    "placeholder": "",
                    "describe": __("task运行时长限制，为0表示不限制(单位s)"),
                    "editable": 1,
                    "addable": 0  # 1 为仅在添加时可修改
                },
                "retry": {
                    "type": "str",
                    "item_type": "str",
                    "label": __("重试次数"),
                    "require": 1,
                    "choice": [],
                    "range": "",
                    "default": '0',
                    "placeholder": "",
                    "describe": __("重试次数"),
                    "editable": 1,
                    "addable": 0
                }
            },
        },
        # 模板分组的排序
        "template_group_order": [__(x) for x in ["绑定任务", "出库入库", "数据计算", "脚本执行"]],
        # 模板列表。
        "templte_list": {
            __("绑定任务"): [
                {
                    "template_name": __("已存在任务"),
                    "templte_ui_config": {
                        __("参数"): {
                            "etl_task_id": {
                                "type": "str",
                                "item_type": "str",
                                "label": __("已存在任务的us task id"),
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "",
                                "placeholder": "",
                                "describe": __("已存在任务的us task id"),
                                "editable": 1
                            }
                        }
                    },
                    "label": __("绑定任务"),
                    "describe": __("绑定已存在任务，类似于创建软链接"),
                    "help_url": conf.get('HELP_URL', {}).get('etl_pipeline', ''),
                    "pass_through": {
                        # 无论什么内容  通过task的字段透传回来
                    }
                },
                {
                    "template_name": __("xx平台任务流"),
                    "templte_ui_config": {
                        __("参数"): {
                            "pipeline_id": {
                                "type": "str",
                                "item_type": "str",
                                "label": __("xx平台任务流id"),
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "",
                                "placeholder": "",
                                "describe": __("xx平台任务流id，可以在任务流详情处查看。"),
                                "editable": 1
                            }
                        }
                    },
                    "label": __("xx平台任务流"),
                    "describe": __("绑定xx平台任务流，类似于创建软链接。用于创建依赖。"),
                    "help_url": conf.get('HELP_URL', {}).get('etl_pipeline', ''),
                    "pass_through": {
                        # 无论什么内容  通过task的字段透传回来
                    }
                }
            ],
            __("出库入库"): [
                {
                    "template_name": __("hdfs入库至hive"),
                    "templte_ui_config": {
                        __("参数"): {
                            "charSet": {
                                "type": "str",
                                "item_type": "str",
                                "label": __("源文件字符集"),
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "UTF-8",
                                "placeholder": "",
                                "describe": __("源文件字符集"),
                                "editable": 1
                            },
                            "databaseName": {
                                "type": "str",
                                "item_type": "str",
                                "label": __("hive数据库名称"),
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "",
                                "placeholder": "",
                                "describe": __("hive数据库名称"),
                                "editable": 1
                            },
                            "tableName": {
                                "type": "str",
                                "item_type": "str",
                                "label": __("hive表名"),
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "",
                                "placeholder": "",
                                "describe": __("hive表名"),
                                "editable": 1
                            },
                            "delimiter": {
                                "type": "str",
                                "item_type": "str",
                                "label": __("源文件分隔符, 填ascii码"),
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "9",
                                "placeholder": "",
                                "describe": __("默认TAB，ascii码：9"),
                                "editable": 1
                            },
                            "failedOnZeroWrited": {
                                "type": "str",
                                "item_type": "str",
                                "label": __("入库为空时任务处理"),
                                "require": 1,
                                "choice": ["1", "0"],
                                "range": "",
                                "default": "1",
                                "placeholder": "",
                                "describe": __("无源文件或入库记录为0时,可以指定任务为成功(0)或失败(1)，默认失败(1)"),
                                "editable": 1
                            },
                            "partitionType": {
                                "type": "str",
                                "item_type": "str",
                                "label": __("分区格式"),
                                "require": 1,
                                "choice": ["P_${YYYYMM}", "P_${YYYYMMDD}", "P_${YYYYMMDDHH}", "NULL"],
                                "range": "",
                                "default": "P_${YYYYMMDDHH}",
                                "placeholder": "",
                                "describe": __("分区格式：P_${YYYYMM}、P_${YYYYMMDD}、P_${YYYYMMDDHH}、NULL"),
                                "editable": 1
                            },
                            "sourceFilePath": {
                                "type": "str",
                                "item_type": "str",
                                "label": __("数据文件hdfs路径"),
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "",
                                "placeholder": "",
                                "describe": __("支持三种日期变量:${YYYYMM}、${YYYYMMDD}、${YYYYMMDDHH}。系统用任务实例的数据时间替换日期变量。"),
                                "editable": 1
                            },
                            "sourceFileNames": {
                                "type": "str",
                                "item_type": "str",
                                "label": __("源文件名"),
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "*",
                                "placeholder": "",
                                "describe": __("源文件名(支持通配符*和${YYYYMMDD});入库不做检查"),
                                "editable": 1
                            },
                            "sourceColumnNames": {
                                "type": "str",
                                "item_type": "str",
                                "label": __("源文件的栏位名称"),
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "",
                                "placeholder": "",
                                "describe": __("源文件的栏位名称，以逗号分割（结尾不能是逗号）,必须保证列数和文件内容一致（创建临时表所用表列名）。例如column1,column2,column3。注：不允许输入空格，源文件栏位名称只由大小写字符、数字和下划线组成"),
                                "editable": 1
                            },
                            "targetColumnNames": {
                                "type": "str",
                                "item_type": "str",
                                "label": __("字段映射关系，即hive表的列名"),
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "",
                                "placeholder": "",
                                "describe": __("字段映射关系，即hive表的列名。"),
                                "editable": 1
                            },
                            "loadMode": {
                                "type": "str",
                                "item_type": "str",
                                "label": __("数据入库模式"),
                                "require": 1,
                                "choice": ["TRUNCATE", "APPEND"],
                                "range": "",
                                "default": "TRUNCATE",
                                "placeholder": "",
                                "describe": __("数据入库模式,TRUNCATE或APPEND;"),
                                "editable": 1
                            }
                        }
                    },
                    "label": __("hdfs入库至hive任务"),
                    "describe": __("hdfs入库至hive任务"),
                    "help_url": conf.get('HELP_URL', {}).get('etl_pipeline', ''),
                    "pass_through": {
                        # 无论什么内容  通过task的字段透传回来
                    }
                },
                {
                    "template_name": __("hive出库至hdfs"),
                    "templte_ui_config": {
                        __("参数"): {
                            "databaseName": {
                                "type": "str",
                                "item_type": "str",
                                "label": __("hive表所在的database"),
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "",
                                "placeholder": "",
                                "describe": __("hive表所在的database"),
                                "editable": 1
                            },
                            "destCheckFileName": {
                                "type": "str",
                                "item_type": "str",
                                "label": __("对账文件名"),
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "",
                                "placeholder": "",
                                "describe": __("对账文件名"),
                                "editable": 1
                            },
                            "destCheckFilePath": {
                                "type": "str",
                                "item_type": "str",
                                "label": __("对账文件路径"),
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "",
                                "placeholder": "",
                                "describe": __("对账文件路径"),
                                "editable": 1
                            },
                            "destFileDelimiter": {
                                "type": "str",
                                "item_type": "str",
                                "label": __("出库文件分隔符"),
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "9",
                                "placeholder": "",
                                "describe": __("出库文件分隔符，填ascii字符对应的数字。默认TAB：9"),
                                "editable": 1
                            },
                            "destFilePath": {
                                "type": "str",
                                "item_type": "str",
                                "label": __("出库文件路径"),
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "",
                                "placeholder": "",
                                "describe": __("出库文件路径"),
                                "editable": 1
                            },
                            "filterSQL": {
                                "type": "text",
                                "item_type": "sql",
                                "label": __("源SQL"),
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": 'select t1,t2,t3 from your_table where imp_date=${YYYYMMDD}',
                                "placeholder": "",
                                "describe": __("源SQL"),
                                "editable": 1
                            }
                        }
                    },
                    "label": __("hive出库至hdfs任务"),
                    "describe": __("hive出库至hdfs任务"),
                    "help_url": conf.get('HELP_URL', {}).get('etl_pipeline', ''),
                    "pass_through": {
                        # 无论什么内容  通过task的字段透传回来
                    }
                },
                {
                    "template_name": __("hdfs导入cos"),
                    "templte_ui_config": {
                        __("参数"): {
                            "hdfsPath": {
                                "type": "str",
                                "item_type": "str",
                                "label": __("hdfs文件路径"),
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "hdfs://xx/xxx",
                                "placeholder": "",
                                "describe": __("源hdfs文件路径，包括文件名，支持通配符*，支持${YYYYMMDD}等的日期变量。如果没hdfs路径权限，联系平台管理员。"),
                                "editable": 1
                            },
                            "cosPath": {
                                "type": "str",
                                "item_type": "str",
                                "label": __("目标cos文件路径"),
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "/xx/xx/${YYYYMMDD}.tar.gz",
                                "placeholder": "",
                                "describe": __("目标cos文件路径，需包括文件名，支持${YYYYMMDD}等的日期变量，如果有多个文件上传，会在自动在cos文件名后面添加一个随机串。"),
                                "editable": 1
                            },
                            "ifNeedZip": {
                                "type": "str",
                                "item_type": "str",
                                "label": __("是否需要压缩"),
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "1",
                                "placeholder": "",
                                "describe": __("是否需要压缩 {0:不需要,1:需要}。压缩会压缩成单个文件。压缩方式为.tar.gz"),
                                "editable": 1
                            }
                        }
                    },
                    "label": __("hdfs导入cos/oss/obs"),
                    "describe": __("hdfs导入cos/oss/obs，基于us的调用shell脚本任务类型实现"),
                    "help_url": conf.get('HELP_URL', {}).get('etl_pipeline', ''),
                    "pass_through": {
                        # 无论什么内容  通过task的字段透传回来
                    }
                },
                {
                    "template_name": __("cos导入hdfs"),
                    "templte_ui_config": {
                        __("参数"): {
                            "hdfsPath": {
                                "type": "str",
                                "item_type": "str",
                                "label": __("hdfs文件路径"),
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "hdfs://xx/xxx",
                                "placeholder": "",
                                "describe": __("目标hdfs文件路径，不包括文件名，支持${YYYYMMDD}等的日期变量。如果没hdfs路径权限，先联系平台管理员。"),
                                "editable": 1
                            },
                            "cosPath": {
                                "type": "str",
                                "item_type": "str",
                                "label": __("源cos文件路径"),
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "/xx/${YYYYMMDD}.tar.gz",
                                "placeholder": "",
                                "describe": __("源cos文件路径，需包括文件名，支持${YYYYMMDD}等的日期变量。如果有多个文件上传，先打成一个.tar.gz压缩包。"),
                                "editable": 1
                            },
                            "ifNeedZip": {
                                "type": "str",
                                "item_type": "str",
                                "label": __("是否需要解压缩"),
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "1",
                                "placeholder": "",
                                "describe": __("是否需要解压缩 {0:不需要,1:需要}。解压方式为tar zcvf。解压后文件会放在目标文件夹。"),
                                "editable": 1
                            }
                        }
                    },
                    "label": __("cos/oss/obs导入hdfs"),
                    "describe": __("cos/oss/obs导入hdfs，基于us的调用shell脚本任务类型实现"),
                    "help_url": conf.get('HELP_URL', {}).get('etl_pipeline', ''),
                    "pass_through": {
                        # 无论什么内容  通过task的字段透传回来
                    }
                },
            ],
            __("数据计算"): [
                {
                    "template_name": __("SparkScala"),
                    "templte_ui_config": {
                        __("参数"): {
                            "jar_path": {
                                "type": "text",
                                "item_type": "str",
                                "label": __("jar包路径"),
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "",
                                "placeholder": "",
                                "describe": __("填写jar包在notebook里的路径，示例/mnt/admin/pipeline_test.py"),
                                "editable": 1
                            },
                            "className": {
                                "type": "str",
                                "item_type": "str",
                                "label": __("jar包中主类的名字"),
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "",
                                "placeholder": "",
                                "describe": __("jar包中主类的名字"),
                                "editable": 1
                            },
                            "files": {
                                "type": "text",
                                "item_type": "str",
                                "label": __("资源文件列表(--files or --archieves)"),
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "",
                                "placeholder": "",
                                "describe": __("暂未支持"),
                                "editable": 1
                            },
                            "programSpecificParams": {
                                "type": "text",
                                "item_type": "str",
                                "label": __("传递给程序的参数"),
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "",
                                "placeholder": "",
                                "describe": __("传递给程序的参数,空格分隔,不要换行"),
                                "editable": 1
                            },
                            "options": {
                                "type": "text",
                                "item_type": "str",
                                "label": __("spark扩展参数"),
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "",
                                "placeholder": "",
                                "describe": __('''
选项（spark-submit的--conf参数)。不带分号，使用换行分隔(例如):
spark.driver.maxResultSize=15G
spark.driver.cores=4
spark支持一系列--conf扩展属性，此处可以直接填写。例如：spark.yarn.am.waitTime=100s。
提交任务时后台会将参数带上提交。换行分隔！！
'''.strip()),
                                "editable": 1
                            },
                            "dynamicAllocation": {
                                "type": "choice",
                                "item_type": "str",
                                "label": __("是否动态资源分配"),
                                "require": 1,
                                "choice": ["1", "0"],
                                "range": "",
                                "default": "1",
                                "placeholder": "",
                                "describe": __("是否动态资源分配，是：1；否：0"),
                                "editable": 1
                            },
                            "driver_memory": {
                                "type": "str",
                                "item_type": "str",
                                "label": __("driver内存大小"),
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "2g",
                                "placeholder": "",
                                "describe": __("driver内存大小"),
                                "editable": 1
                            },
                            "num_executors": {
                                "type": "str",
                                "item_type": "str",
                                "label": __("executor数量"),
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "4",
                                "placeholder": "",
                                "describe": __("executor数量"),
                                "editable": 1
                            },
                            "executor_memory": {
                                "type": "str",
                                "item_type": "str",
                                "label": __("executor内存大小"),
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "2g",
                                "placeholder": "",
                                "describe": __("executor内存大小"),
                                "editable": 1
                            },

                            "executor_cores": {
                                "type": "str",
                                "item_type": "str",
                                "label": __("executor核心数"),
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "2",
                                "placeholder": "",
                                "describe": __("executor核心数"),
                                "editable": 1
                            },

                            "task.main.timeout": {
                                "type": "str",
                                "item_type": "str",
                                "label": __("超时时间，单位分钟"),
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "480",
                                "placeholder": "",
                                "describe": __("超时时间，单位分钟：480 (代表8小时)"),
                                "editable": 1
                            },
                            "task.check.timeout": {
                                "type": "str",
                                "item_type": "str",
                                "label": __("check超时时间，单位分钟"),
                                "require": 1,
                                "choice": ["5", "10"],
                                "range": "",
                                "default": "5",
                                "placeholder": "",
                                "describe": __("check超时时间，单位分钟"),
                                "editable": 1
                            }
                        }
                    },
                    "label": __("SparkScala"),
                    "describe": __("SparkScala计算"),
                    "help_url": conf.get('HELP_URL', {}).get('etl_pipeline', ''),
                    "pass_through": {
                        # 无论什么内容  通过task的字段透传回来
                    }
                },
                {
                    "template_name": __("SQL"),
                    "templte_ui_config": {
                        __("参数"): {
                            "filterSQL": {
                                "type": "text",
                                "item_type": "sql",
                                "label": __("计算加工逻辑"),
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": __('''
--库名，替换下面的demo_database
use demo_database;

--建表语句，替换下面的demo_table，修改字段。一定要加“if not exists”，这样使只在第一次运行时建表
CREATE TABLE if not exists demo_table(
    qimei36 STRING COMMENT '唯一设备ID',
    userid_id STRING COMMENT '用户id（各app的用户id）',
    device_id STRING COMMENT '设备id（各app的device_id）',
    ftime INT COMMENT '数据分区时间 格式：yyyymmdd'
)
PARTITION BY LIST( ftime )          --定义分区字段，替换掉ftime。
(
    PARTITION p_20220323 VALUES IN ( 20220323 ),       --初始分区，分区名替换p_20220323，分区值替换20220323
    PARTITION default
)
STORED AS ORCFILE COMPRESS;

-- 分区，根据时间参数新建分区。
alter table demo_table drop partition (p_${YYYYMMDD});
alter table demo_table add partition p_${YYYYMMDD} values in (${YYYYMMDD});

-- 写入，用你的sql逻辑替换。
insert table demo_table
select * from other_db::other_table partition(p_${YYYYMMDD}) t;
'''),
                                "placeholder": "",
                                "describe": __("从hive导出数据的sql，比如 select a,b,c FROM table where imp_date='${YYYYMMDD}' ;sql末尾不要用分号结尾"),
                                "editable": 1
                            },

                            "special_para": {
                                "type": "str",
                                "item_type": "str",
                                "label": __("hive特殊参数"),
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "set hive.exec.parallel = true;set hive.execute.engine=spark;set hive.multi.join.use.hive=false;set hive.spark.failed.retry=false;",
                                "placeholder": "",
                                "describe": __("hive特殊参数"),
                                "editable": 1
                            }
                        }
                    },
                    "label": __("sql执行"),
                    "describe": __("sql执行"),
                    "help_url": conf.get('HELP_URL', {}).get('etl_pipeline', ''),
                    "pass_through": {
                        # 无论什么内容  通过task的字段透传回来
                    }
                },
                {
                    "template_name": __("pyspark"),
                    "templte_ui_config": {
                        __("参数"): {
                            "py_script_path": {
                                "type": "str",
                                "item_type": "str",
                                "label": __("pyspark脚本路径"),
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "",
                                "placeholder": "",
                                "describe": __("填写pyspark脚本在notebook里的路径，示例/mnt/admin/pipeline_test.py"),
                                "editable": 1
                            },
                            "files": {
                                "type": "str",
                                "item_type": "str",
                                "label": __("资源文件列表"),
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "",
                                "placeholder": "",
                                "describe": __("暂未支持"),
                                "editable": 1
                            },
                            "pyFiles": {
                                "type": "str",
                                "item_type": "str",
                                "label": __("执行脚本依赖文件列表(spark-submit中的--py-files)"),
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "",
                                "placeholder": "",
                                "describe": __("暂未支持"),
                                "editable": 1
                            },
                            "programSpecificParams": {
                                "type": "text",
                                "item_type": "str",
                                "label": __("传递给程序的参数"),
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "",
                                "placeholder": "",
                                "describe": __("传递给程序的参数,空格分隔,不要换行"),
                                "editable": 1
                            },
                            "options": {
                                "type": "text",
                                "item_type": "str",
                                "label": __("spark扩展参数"),
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": '',
                                "placeholder": "",
                                "describe": __("选项（spark-submit的--conf参数)。例如：spark.yarn.am.waitTime=100s。换行分隔！！"),
                                "editable": 1
                            },
                            "dynamicAllocation": {
                                "type": "str",
                                "item_type": "str",
                                "label": __("是否动态资源分配"),
                                "require": 1,
                                "choice": [1, 0],
                                "range": "",
                                "default": 1,
                                "placeholder": "",
                                "describe": __("是否动态资源分配，是：1；否：0"),
                                "editable": 1
                            },
                            "driver_memory": {
                                "type": "str",
                                "item_type": "str",
                                "label": __("driver内存大小"),
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "2g",
                                "placeholder": "",
                                "describe": __("driver内存大小"),
                                "editable": 1
                            },
                            "num_executors": {
                                "type": "str",
                                "item_type": "str",
                                "label": __("executor数量"),
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": 4,
                                "placeholder": "",
                                "describe": __("executor数量"),
                                "editable": 1
                            },

                            "executor_memory": {
                                "type": "str",
                                "item_type": "str",
                                "label": __("executor内存大小"),
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "2g",
                                "placeholder": "",
                                "describe": __("executor内存大小"),
                                "editable": 1
                            },

                            "executor_cores": {
                                "type": "str",
                                "item_type": "str",
                                "label": __("executor核心数"),
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": 2,
                                "placeholder": "",
                                "describe": __("executor核心数"),
                                "editable": 1
                            },

                            "task.main.timeout": {
                                "type": "str",
                                "item_type": "str",
                                "label": __("超时时间，单位分钟"),
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": 480,
                                "placeholder": "",
                                "describe": __("超时时间，单位分钟：480 (代表8小时)"),
                                "editable": 1
                            },
                            "task.check.timeout": {
                                "type": "int",
                                "item_type": "int",
                                "label": __("check超时时间，单位分钟"),
                                "require": 1,
                                "choice": ["5", "10"],
                                "range": "",
                                "default": "5",
                                "placeholder": "",
                                "describe": __("check超时时间，单位分钟"),
                                "editable": 1
                            }
                        }
                    },
                    "label": __("pyspark"),
                    "describe": __("pyspark脚本执行"),
                    "help_url": conf.get('HELP_URL', {}).get('etl_pipeline', ''),
                    "pass_through": {
                        # 无论什么内容  通过task的字段透传回来
                    }
                }
            ],
            __("脚本执行"): [
                {
                    "template_name": __("shell"),
                    "templte_ui_config": {
                        __("参数"): {
                            "command": {
                                "type": "text",
                                "item_type": "str",
                                "label": __("执行命令"),
                                "require": 1,
                                "choice": [],
                                "range": "",
                                "default": "",
                                "placeholder": "",
                                "describe": __("执行的命令"),
                                "editable": 1
                            }
                        }
                    },
                    "label": __("shell"),
                    "describe": __("shell脚本执行"),
                    "help_url": conf.get('HELP_URL', {}).get('etl_pipeline', ''),
                    "pass_through": {
                        # 无论什么内容  通过task的字段透传回来
                    }
                }
            ]
        },
        "status": 0
    }
    # todo: pipeline的公共参数配置
    pipeline_config_ui = {
        "alert": {
            "alert_user": {
                "type": "str",
                "item_type": "str",
                "label": __("任务流负责人"),
                "require": 1,
                "choice": [],
                "range": "",
                "default": "",
                "placeholder": __("任务流负责人，逗号分隔"),
                "describe": __("任务流负责人，逗号分隔。会添加到每个任务的负责人。"),
                "editable": 1
            }
        }
    }

    # @property
    def pipeline_jump_button(self):
        button = [
            {
                "name": _("调度实例"),
                "action_url": self.host + "/tree?dag_id=%s" % self.pipeline.name,
                "icon_svg": '<svg t="1660554835088" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="2435" width="200" height="200"><path d="M112.64 95.36a32 32 0 0 0-32 32v332.16a32 32 0 0 0 32 32h332.16a32 32 0 0 0 32-32V128a32 32 0 0 0-32-32z m300.16 332.16H144.64V159.36h268.16zM938.88 293.76a197.76 197.76 0 1 0-197.76 197.76 198.4 198.4 0 0 0 197.76-197.76z m-332.16 0a133.76 133.76 0 1 1 133.76 133.76 134.4 134.4 0 0 1-133.76-133.76zM99.84 928.64h365.44a32 32 0 0 0 27.52-48L310.4 563.84a33.28 33.28 0 0 0-55.68 0l-182.4 316.8a32 32 0 0 0 27.52 48z m182.4-284.16l128 220.16h-256zM832 552.96h-177.28a32 32 0 0 0-27.52 16l-89.6 155.52a32 32 0 0 0 0 32l89.6 155.52a32 32 0 0 0 27.52 16H832a32 32 0 0 0 27.52-16l89.6-155.52a32 32 0 0 0 0-32l-89.6-155.52a32 32 0 0 0-27.52-16z m-18.56 311.04h-140.16L601.6 741.12l71.68-123.52h142.72l71.68 123.52z" fill="#225ed2" p-id="2436"></path></svg>'
            },
            {
                "name": __("日志"),
                "action_url": self.host + "/graph?dag_id=%s" % self.pipeline.name,
                "icon_svg": '<svg t="1669023206369" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="2716" width="64" height="64"><path d="M269.844659 81.4308h44.821057v166.626082h-44.821057zM677.140966 491.719232c52.335426 0 102.092273 19.937769 140.105639 56.13883 38.126482 36.31053 60.461599 85.284073 62.891788 137.900467 2.5056 54.276658-16.27424 106.280032-52.881549 146.431672-36.60731 40.15164-86.65972 63.643469-140.936379 66.150285-3.180653 0.147174-6.401444 0.221369-9.576016 0.221369-52.341508 0-102.102004-19.936552-140.114153-56.136398-38.126482-36.309314-60.461599-85.284073-62.891789-137.902899-2.5056-54.276658 16.27424-106.280032 52.88155-146.431672 36.60731-40.15164 86.65972-63.643469 140.936379-66.149069a208.122961 208.122961 0 0 1 9.576016-0.221369h0.008514m-0.00973-44.822274c-3.859355 0-7.746684 0.088791-11.642528 0.268805-136.951744 6.3236-242.847422 122.470346-236.525038 259.422091 6.143586 133.0559 115.942406 236.793842 247.779562 236.793842 3.859355 0 7.747901-0.088791 11.642529-0.268804 136.951744-6.322384 242.847422-122.470346 236.525037-259.422091-6.143586-133.057117-115.942406-236.798708-247.779562-236.793843z" p-id="2717" fill="#305FCB"></path><path d="M490.264524 891.110734a272.361206 272.361206 0 0 1-32.682275-37.369937H180.453104c-20.912034 0-37.927007-17.013757-37.927007-37.92579v-590.263526c0-20.912034 17.013757-37.927007 37.927007-37.927007H732.799354c20.912034 0 37.925791 17.013757 37.925791 37.927007V441.15597a268.605238 268.605238 0 0 1 44.821057 21.463023V225.551481c0-45.70045-37.047614-82.746848-82.746848-82.746849H180.453104c-45.70045 0-82.746848 37.047614-82.746848 82.746849v590.263526c0 45.70045 37.047614 82.746848 82.746848 82.746848h317.980164a273.587248 273.587248 0 0 1-8.168744-7.451121z" p-id="2718" fill="#305FCB"></path><path d="M770.725145 489.61623a225.243754 225.243754 0 0 1 44.821057 27.231985v-0.21407a225.182938 225.182938 0 0 0-44.821057-27.114003v0.096088zM812.590566 778.530212H646.820768V576.105667h44.821057v157.604704h120.948741zM209.55091 380.121489h498.255687v44.821057H209.55091zM600.682445 81.4308h44.821058v166.626082h-44.821058zM406.842623 712.17437H209.55091v44.821057h203.864657a272.351476 272.351476 0 0 1-6.572944-44.821057zM450.941192 546.147929H209.55091v44.821057h217.435038a268.707408 268.707408 0 0 1 23.955244-44.821057z" p-id="2719" fill="#305FCB"></path></svg>'
            }
        ]
        # print(button)
        return button

    # @property
    def pipeline_run_button(self):
        button = [
            {
                "name": __("提交"),
                "action_url": "/etl_pipeline_modelview/submit_etl_pipeline/%s" % self.pipeline.id,
                "icon_svg": '<svg t="1660558913467" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="5032" width="200" height="200"><path d="M689.3568 820.9408 333.6192 820.9408c-13.9264 0-25.1904-11.264-25.1904-25.1904L308.4288 465.152l-211.968 0c-10.1888 0-19.4048-6.144-23.296-15.5648C69.2224 440.2176 71.424 429.312 78.6432 422.144l415.0272-415.0784c9.472-9.472 26.1632-9.472 35.6352 0l411.392 411.3408c7.2704 4.4544 12.0832 12.4416 12.0832 21.5552 0 14.1312-11.52 24.576-25.7024 25.1904-0.1536 0-0.3072 0-0.512 0l-211.968 0 0 330.5472C714.5984 809.6256 703.2832 820.9408 689.3568 820.9408zM358.8096 770.5088l305.3568 0L664.1664 439.9616c0-13.9264 11.264-25.1904 25.1904-25.1904l176.3328 0L511.488 60.5184 157.2864 414.7712l176.3328 0c13.9264 0 25.1904 11.264 25.1904 25.1904L358.8096 770.5088z" p-id="5033" fill="#225ed2"></path><path d="M96.4096 923.1872l830.1056 0L926.5152 1024 96.4096 1024 96.4096 923.1872 96.4096 923.1872zM96.4096 923.1872" p-id="5034" fill="#225ed2"></path></svg>'
            }
        ]
        # print(button)
        return button

    # todo 任务流编排 运行按钮触发函数
    # @pysnooper.snoop(watch_explode=())
    def submit_pipeline(self):
        # todo 检查任务是否存在，提交创建新的任务或修改旧任务，或者删除任务
        # todo 保存到调度平台，并发起远程调度
        return "", self.host + "/code?dag_id=%s" % self.pipeline.name

    # todo: 删除前先把下面的task删除了
    # @pysnooper.snoop()
    def delete_pipeline(self):
        # 删除远程上下游关系和远程任务
        pass

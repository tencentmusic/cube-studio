import ajax from './ajax';
import { IPipelineAdd, IPipelineEdit } from '../types/pipeline';
import { ITaskAdd, ITaskEdit } from '../types/task';

// const getTemplateCommandConfig = (): Promise<any> => {
//   return ajax.get('/etl_pipeline_modelview/template/list/');
// };
const getTemplateCommandConfig = (pipelineId: number | string): Promise<any> => {
  return ajax.get(`/etl_pipeline_modelview/template/list/${pipelineId}`);
  return new Promise((resolve, reject) => {
    const cusRes = {
      "message": "success",
      "templte_common_ui_config": {
        "crontab": {
          "type": "str",
          "item_type": "str",
          "label": "调度周期",
          "require": 1,
          "choice": [],
          "range": "",
          "default": "",
          "placeholder": "",
          "describe": "周期任务的时间设定 * * * * * 表示为 minute hour day month week",
          "editable": 1
        },
        "selfDepend": {
          "type": "str",
          "item_type": "str",
          "label": "自依赖判断",
          "require": 1,
          "choice": [
            "自依赖",
            "单实例运行",
            "多实例运行"
          ],
          "range": "",
          "default": "单实例运行",
          "placeholder": "",
          "describe": "一个任务的多次调度实例之间是否要进行前后依赖",
          "editable": 1
        }
      },
      "templte_list": {
        "出库入库": [
          {
            "template_name": "导入",
            "templte_ui_config": {
              "shell": {
                "sourceServer": {
                  "type": "str",
                  "item_type": "str",
                  "label": "源服务器",
                  "require": 1,
                  "choice": [],
                  "range": "",
                  "default": "3",
                  "placeholder": "",
                  "describe": "源服务器 ，例如：tdw出库mysql 源服务器就是 TDW集群",
                  "editable": 1
                },
                "targetServer": {
                  "type": "str",
                  "item_type": "str",
                  "label": "目标服务器",
                  "require": 1,
                  "choice": [],
                  "range": "",
                  "default": "3",
                  "placeholder": "",
                  "describe": "目标服务器（注意是目标服务器名，不是IP/域名），例如：tdw出库mysql 目标服务器就是 MySQL数据库，这里只填写服务器引用，配置需要到安全中心:https://security.tianqiong.woa.com/auth/group",
                  "editable": 1
                },
                "charSet": {
                  "type": "str",
                  "item_type": "str",
                  "label": "源文件字符集",
                  "require": 1,
                  "choice": [],
                  "range": "",
                  "default": "UTF-8",
                  "placeholder": "",
                  "describe": "源文件字符集",
                  "editable": 1
                },
                "databaseName": {
                  "type": "str",
                  "item_type": "str",
                  "label": "DB名称",
                  "require": 1,
                  "choice": [],
                  "range": "",
                  "default": "",
                  "placeholder": "",
                  "describe": "DB名称",
                  "editable": 1
                },
                "delimiter": {
                  "type": "str",
                  "item_type": "str",
                  "label": "源文件分隔符, 填ascii码",
                  "require": 1,
                  "choice": [],
                  "range": "",
                  "default": "9",
                  "placeholder": "",
                  "describe": "默认TAB，ascii码：9",
                  "editable": 1
                },
                "failedOnZeroWrited": {
                  "type": "str",
                  "item_type": "str",
                  "label": "入库为空时任务处理,无源文件或入库记录为0时,可以指定任务为成功(0)或失败(1)",
                  "require": 1,
                  "choice": [],
                  "range": "",
                  "default": "1",
                  "placeholder": "",
                  "describe": "默认失败(1)",
                  "editable": 1
                },
                "partitionType": {
                  "type": "str",
                  "item_type": "str",
                  "label": "分区格式",
                  "require": 1,
                  "choice": [],
                  "range": "",
                  "default": "P_${YYYYMMDDHH}",
                  "placeholder": "",
                  "describe": "分区格式：P_${YYYYMM},P_${YYYYMMDD},P_${YYYYMMDDHH}",
                  "editable": 1
                },
                "sourceFilePath": {
                  "type": "str",
                  "item_type": "str",
                  "label": "数据文件路径",
                  "require": 1,
                  "choice": [],
                  "range": "",
                  "default": "",
                  "placeholder": "",
                  "describe": "支持三种日期变量:${YYYYMM}，${YYYYMMDD}，${YYYYMMDDHH}。系统用任务实例的数据时间替换日期变量。",
                  "editable": 1
                },
                "sourceFileNames": {
                  "type": "str",
                  "item_type": "str",
                  "label": "源文件名",
                  "require": 1,
                  "choice": [],
                  "range": "",
                  "default": "",
                  "placeholder": "",
                  "describe": "源文件名(支持通配符*和${YYYYMMDD});入库不做检查",
                  "editable": 1
                },
                "sourceColumnNames": {
                  "type": "str",
                  "item_type": "str",
                  "label": "源文件的栏位名称",
                  "require": 1,
                  "choice": [],
                  "range": "",
                  "default": "",
                  "placeholder": "",
                  "describe": "源文件的栏位名称，以逗号分割（结尾不能是逗号）,必须保证列数和文件内容一致（创建临时表所用表列名）。例如column1,column2,column3。注：不允许输入空格，源文件栏位名称只由大小写字符、数字和下划线组成",
                  "editable": 1
                },
                "tableName": {
                  "type": "str",
                  "item_type": "str",
                  "label": "TDW数据库名",
                  "require": 1,
                  "choice": [],
                  "range": "",
                  "default": "",
                  "placeholder": "",
                  "describe": "TDW数据库名。",
                  "editable": 1
                },
                "targetColumnNames": {
                  "type": "str",
                  "item_type": "str",
                  "label": "字段映射关系，即tdw表的列名",
                  "require": 1,
                  "choice": [],
                  "range": "",
                  "default": "",
                  "placeholder": "",
                  "describe": "字段映射关系，即tdw表的列名。",
                  "editable": 1
                },
                "writeConcurrency": {
                  "type": "str",
                  "item_type": "str",
                  "label": "写结果库的并发session数",
                  "require": 1,
                  "choice": [],
                  "range": "",
                  "default": "3",
                  "placeholder": "",
                  "describe": "写结果库的并发session数,取值为：1,2,4，三个取值。",
                  "editable": 1
                }
              }
            },
            "username": "uthermai",
            "changed_on": "2022-02-17 07:01:54",
            "created_on": "2022-02-17 07:01:54",
            "label": "HDFS入库至TDW任务",
            "describe": "HDFS入库至TDW任务",
            "help_url": "",
            "pass_through": {},
            "template_id": 1
          },
          {
            "template_name": "导出",
            "templte_ui_config": {
              "shell": {
                "sourceServer": {
                  "type": "str",
                  "item_type": "str",
                  "label": "源服务器",
                  "require": 1,
                  "choice": [],
                  "range": "",
                  "default": "3",
                  "placeholder": "",
                  "describe": "源服务器 ，例如：tdw出库mysql 源服务器就是 TDW集群",
                  "editable": 1
                },
                "targetServer": {
                  "type": "str",
                  "item_type": "str",
                  "label": "目标服务器",
                  "require": 1,
                  "choice": [],
                  "range": "",
                  "default": "3",
                  "placeholder": "",
                  "describe": "目标服务器（注意是目标服务器名，不是IP/域名），例如：tdw出库mysql 目标服务器就是 MySQL数据库，这里只填写服务器引用，配置需要到安全中心:https://security.tianqiong.woa.com/auth/group",
                  "editable": 1
                },
                "databaseName": {
                  "type": "str",
                  "item_type": "str",
                  "label": "TDW表所在的database",
                  "require": 1,
                  "choice": [],
                  "range": "",
                  "default": "",
                  "placeholder": "",
                  "describe": "TDW表所在的database",
                  "editable": 1
                },
                "destCheckFileName": {
                  "type": "str",
                  "item_type": "str",
                  "label": "对账文件名",
                  "require": 1,
                  "choice": [],
                  "range": "",
                  "default": "",
                  "placeholder": "",
                  "describe": "对账文件名",
                  "editable": 1
                },
                "destCheckFilePath": {
                  "type": "str",
                  "item_type": "str",
                  "label": "对账文件路径",
                  "require": 1,
                  "choice": [],
                  "range": "",
                  "default": "",
                  "placeholder": "",
                  "describe": "对账文件路径",
                  "editable": 1
                },
                "destFileDelimiter": {
                  "type": "str",
                  "item_type": "str",
                  "label": "出库文件分隔符",
                  "require": 1,
                  "choice": [],
                  "range": "",
                  "default": "9",
                  "placeholder": "",
                  "describe": "出库文件分隔符，填ascii字符对应的数字。默认TAB：9",
                  "editable": 1
                },
                "destFilePath": {
                  "type": "str",
                  "item_type": "str",
                  "label": "出库文件路径",
                  "require": 1,
                  "choice": [],
                  "range": "",
                  "default": "",
                  "placeholder": "",
                  "describe": "出库文件路径",
                  "editable": 1
                },
                "filterSQL": {
                  "type": "str",
                  "item_type": "str",
                  "label": "源SQL",
                  "require": 1,
                  "choice": [],
                  "range": "",
                  "default": "",
                  "placeholder": "",
                  "describe": "源SQL",
                  "editable": 1
                }
              }
            },
            "username": "uthermai",
            "changed_on": "2022-02-17 07:01:54",
            "created_on": "2022-02-17 07:01:54",
            "label": "TDW出库至HDFS任务",
            "describe": "TDW出库至HDFS任务",
            "help_url": "",
            "pass_through": {},
            "template_id": 2
          }
        ],
        "数据计算": [
          {
            "template_name": "SQL",
            "templte_ui_config": {
              "shell": {
                "filterSQL": {
                  "type": "text",
                  "item_type": "str",
                  "label": "从TDW导出数据的sql",
                  "require": 1,
                  "choice": [],
                  "range": "",
                  "default": "",
                  "placeholder": "",
                  "describe": "从TDW导出数据的sql，比如 select a,b,c FROM table where imp_date='${YYYYMMDD}' ;sql末尾不要用分号结尾",
                  "editable": 1
                },
                "special_para": {
                  "type": "str",
                  "item_type": "str",
                  "label": "tdw特殊参数",
                  "require": 1,
                  "choice": [],
                  "range": "",
                  "default": "set hive.exec.parallel = true;set hive.execute.engine=spark;set hive.multi.join.use.hive=false;set hive.spark.failed.retry=false;",
                  "placeholder": "",
                  "describe": "tdw特殊参数",
                  "editable": 1
                }
              }
            },
            "username": "uthermai",
            "changed_on": "2022-02-17 07:01:54",
            "created_on": "2022-02-17 07:01:54",
            "label": "pythonsql执行",
            "describe": "pythonsql执行",
            "help_url": "",
            "pass_through": {},
            "template_id": 3
          },
          {
            "template_name": "pythonSQL",
            "templte_ui_config": {
              "shell": {
                "file_path": {
                  "type": "str",
                  "item_type": "str",
                  "label": "脚本路径",
                  "require": 1,
                  "choice": [],
                  "range": "",
                  "default": "",
                  "placeholder": "脚本路径",
                  "describe": "脚本路径",
                  "editable": 1
                },
                "params": {
                  "type": "str",
                  "item_type": "str",
                  "label": "脚本参数",
                  "require": 1,
                  "choice": [],
                  "range": "",
                  "default": "",
                  "placeholder": "脚本参数",
                  "describe": "脚本参数",
                  "editable": 1
                },
                "special_para": {
                  "type": "str",
                  "item_type": "str",
                  "label": "TDW参数",
                  "require": 1,
                  "choice": [],
                  "range": "",
                  "default": "set hive.exec.parallel = true;set hive.execute.engine=spark;set hive.multi.join.use.hive=false;set hive.spark.failed.retry=false;",
                  "placeholder": "",
                  "describe": "TDW参数",
                  "editable": 1
                }
              }
            },
            "username": "uthermai",
            "changed_on": "2022-02-17 07:01:54",
            "created_on": "2022-02-17 07:01:54",
            "label": "pythonsql执行",
            "describe": "pythonsql执行",
            "help_url": "",
            "pass_through": {},
            "template_id": 4
          },
          {
            "template_name": "pyspark",
            "templte_ui_config": {
              "shell": {
                "pyScript": {
                  "type": "str",
                  "item_type": "str",
                  "label": "pyspark脚本名称",
                  "require": 1,
                  "choice": [],
                  "range": "",
                  "default": "",
                  "placeholder": "",
                  "describe": "pyspark脚本名称",
                  "editable": 1
                },
                "programSpecificParams": {
                  "type": "str",
                  "item_type": "str",
                  "label": "传递给程序的参数,空格分隔,不要换行",
                  "require": 1,
                  "choice": [],
                  "range": "",
                  "default": "",
                  "placeholder": "",
                  "describe": "传递给程序的参数,空格分隔,不要换行",
                  "editable": 1
                },
                "driver_memory": {
                  "type": "int",
                  "item_type": "int",
                  "label": "driver内存大小",
                  "require": 1,
                  "choice": [],
                  "range": "",
                  "default": "2g",
                  "placeholder": "",
                  "describe": "driver内存大小",
                  "editable": 1
                },
                "num_executors": {
                  "type": "int",
                  "item_type": "int",
                  "label": "executor数量",
                  "require": 1,
                  "choice": [],
                  "range": "",
                  "default": "4",
                  "placeholder": "",
                  "describe": "executor数量",
                  "editable": 1
                },
                "executor_memory": {
                  "type": "int",
                  "item_type": "int",
                  "label": "executor内存大小",
                  "require": 1,
                  "choice": [],
                  "range": "",
                  "default": "2g",
                  "placeholder": "",
                  "describe": "executor内存大小",
                  "editable": 1
                },
                "executor_cores": {
                  "type": "int",
                  "item_type": "int",
                  "label": "executor核心数",
                  "require": 1,
                  "choice": [],
                  "range": "",
                  "default": "2",
                  "placeholder": "",
                  "describe": "executor核心数",
                  "editable": 1
                },
                "task.main.timeout": {
                  "type": "int",
                  "item_type": "int",
                  "label": "超时时间，单位分钟：480 (代表8小时)",
                  "require": 1,
                  "choice": [],
                  "range": "",
                  "default": "480",
                  "placeholder": "",
                  "describe": "超时时间，单位分钟：480 (代表8小时)",
                  "editable": 1
                }
              }
            },
            "username": "uthermai",
            "changed_on": "2022-02-17 07:01:54",
            "created_on": "2022-02-17 07:01:54",
            "label": "pyspark",
            "describe": "pyspark脚本执行",
            "help_url": "",
            "pass_through": {},
            "template_id": 5
          },
          {
            "template_name": "SparkScala",
            "templte_ui_config": {
              "shell": {
                "mapred.jar": {
                  "type": "str",
                  "item_type": "str",
                  "label": "jar包的包名",
                  "require": 1,
                  "choice": [],
                  "range": "",
                  "default": "",
                  "placeholder": "",
                  "describe": "jar包的包名",
                  "editable": 1
                },
                "className": {
                  "type": "str",
                  "item_type": "str",
                  "label": "主类的名字",
                  "require": 1,
                  "choice": [],
                  "range": "",
                  "default": "",
                  "placeholder": "",
                  "describe": "主类的名字",
                  "editable": 1
                },
                "programSpecificParams": {
                  "type": "str",
                  "item_type": "str",
                  "label": "传递给程序的参数,空格分隔,不要换行",
                  "require": 1,
                  "choice": [],
                  "range": "",
                  "default": "",
                  "placeholder": "",
                  "describe": "传递给程序的参数,空格分隔,不要换行",
                  "editable": 1
                },
                "options": {
                  "type": "str",
                  "item_type": "str",
                  "label": "选项（spark支持的选项)。不带分号，使用换行分隔",
                  "require": 1,
                  "choice": [],
                  "range": "",
                  "default": "",
                  "placeholder": "",
                  "describe": "选项（spark支持的选项)。不带分号，使用换行分隔",
                  "editable": 1
                },
                "driver_memory": {
                  "type": "int",
                  "item_type": "int",
                  "label": "driver内存大小",
                  "require": 1,
                  "choice": [],
                  "range": "",
                  "default": "2g",
                  "placeholder": "",
                  "describe": "driver内存大小",
                  "editable": 1
                },
                "num_executors": {
                  "type": "int",
                  "item_type": "int",
                  "label": "executor数量",
                  "require": 1,
                  "choice": [],
                  "range": "",
                  "default": "4",
                  "placeholder": "",
                  "describe": "executor数量",
                  "editable": 1
                },
                "executor_memory": {
                  "type": "int",
                  "item_type": "int",
                  "label": "executor内存大小",
                  "require": 1,
                  "choice": [],
                  "range": "",
                  "default": "2g",
                  "placeholder": "",
                  "describe": "executor内存大小",
                  "editable": 1
                },
                "executor_cores": {
                  "type": "int",
                  "item_type": "int",
                  "label": "executor核心数",
                  "require": 1,
                  "choice": [],
                  "range": "",
                  "default": "2",
                  "placeholder": "",
                  "describe": "executor核心数",
                  "editable": 1
                },
                "task.main.timeout": {
                  "type": "int",
                  "item_type": "int",
                  "label": "超时时间，单位分钟：480 (代表8小时)",
                  "require": 1,
                  "choice": [],
                  "range": "",
                  "default": "480",
                  "placeholder": "",
                  "describe": "超时时间，单位分钟：480 (代表8小时)",
                  "editable": 1
                }
              }
            },
            "username": "uthermai",
            "changed_on": "2022-02-17 07:01:54",
            "created_on": "2022-02-17 07:01:54",
            "label": "SparkScala",
            "describe": "SparkScala脚本执行",
            "help_url": "",
            "pass_through": {},
            "template_id": 6
          }
        ],
        "其他": [
          {
            "template_name": "shell",
            "templte_ui_config": {
              "shell": {
                "file_path": {
                  "type": "str",
                  "item_type": "str",
                  "label": "脚本路径",
                  "require": 1,
                  "choice": [],
                  "range": "",
                  "default": "",
                  "placeholder": "",
                  "describe": "脚本路径",
                  "editable": 1
                },
                "params": {
                  "type": "str",
                  "item_type": "str",
                  "label": "脚本参数",
                  "require": 1,
                  "choice": [],
                  "range": "",
                  "default": "",
                  "placeholder": "",
                  "describe": "脚本参数",
                  "editable": 1
                }
              }
            },
            "username": "uthermai",
            "changed_on": "2022-02-17 07:01:54",
            "created_on": "2022-02-17 07:01:54",
            "label": "shell执行",
            "describe": "shell执行",
            "help_url": "",
            "pass_through": {},
            "template_id": 7
          },
          {
            "template_name": "test",
            "templte_ui_config": {
              "shell": {
                "args1": {
                  "type": "json",
                  "item_type": "str",
                  "label": "参数1",
                  "require": 1,
                  "choice": [],
                  "range": "",
                  "default": "",
                  "placeholder": "",
                  "describe": "参数1",
                  "editable": 1
                },
                "args2": {
                  "type": "str",
                  "item_type": "str",
                  "label": "参数2",
                  "require": 1,
                  "choice": [],
                  "range": "",
                  "default": "",
                  "placeholder": "",
                  "describe": "参数2",
                  "editable": 1
                },
                "args3": {
                  "type": "int",
                  "item_type": "str",
                  "label": "参数3",
                  "require": 1,
                  "choice": [],
                  "range": "",
                  "default": "",
                  "placeholder": "",
                  "describe": "参数3",
                  "editable": 1
                },
                "args4": {
                  "type": "choice",
                  "item_type": "str",
                  "label": "参数4",
                  "require": 1,
                  "choice": [
                    "aa",
                    "bb",
                    "cc",
                    "dd"
                  ],
                  "range": "",
                  "default": "aa",
                  "placeholder": "",
                  "describe": "参数4",
                  "editable": 1
                },
                "args5": {
                  "type": "str",
                  "item_type": "str",
                  "label": "参数5",
                  "require": 1,
                  "choice": [],
                  "range": "",
                  "default": "这是个不可编辑参数",
                  "placeholder": "",
                  "describe": "这个参数不可编辑",
                  "editable": 0
                },
                "args6": {
                  "type": "text",
                  "item_type": "str",
                  "label": "参数6",
                  "require": 1,
                  "choice": [],
                  "range": "",
                  "default": "",
                  "placeholder": "",
                  "describe": "参数6，多行的文本编辑器",
                  "editable": 1
                }
              }
            },
            "username": "uthermai",
            "changed_on": "2022-02-17 07:01:54",
            "created_on": "2022-02-17 07:01:54",
            "label": "模板测试",
            "describe": "模板测试",
            "help_url": "",
            "pass_through": {},
            "template_id": 8
          }
        ]
      },
      "status": 0
    }
    resolve(cusRes)
  })
};
// 保存配置
const pipeline_modelview_save = (pipelineId: number | string, data: Record<any, any>): Promise<any> => {
  return ajax.post({
    url: `/etl_pipeline_modelview/config/${pipelineId}`,
    data,
  });
};

// 获取任务模板列表
const job_template_modelview = (): Promise<any> => {
  return ajax.get('/job_template_modelview/api/');
};

const project_all = (): Promise<any> => {
  return ajax.get('/project_modelview/api/');
};
// 获取 org 项目组
const project_modelview = (): Promise<any> => {
  return ajax.get('/project_modelview/org/api/');
};

// 新增流水线
const pipeline_modelview_add = (data: IPipelineAdd): Promise<any> => {
  return ajax.post({ url: '/pipeline_modelview/api/', data });
};

// 获取流水线列表
const pipeline_modelview_demo = (): Promise<any> => {
  return ajax.get('/pipeline_modelview/demo/list/');
};

// 获取流水线列表
const pipeline_modelview_list = (): Promise<any> => {
  return ajax.get('/pipeline_modelview/my/list/');
};

const pipeline_modelview_all = (filters: string): Promise<any> => {
  return ajax.get(`/pipeline_modelview/api/?form_data=${filters}`);
};

// 获取流水线信息
const pipeline_modelview_detail = (pipelineId: number | string): Promise<any> => {
  return ajax.get(`/etl_pipeline_modelview/config/${pipelineId}`);
  return ajax.get(`/pipeline_modelview/api/${pipelineId}`);
};

// 删除指定流水线
const pipeline_modelview_delete = (pipelineId: number | string): Promise<any> => {
  return ajax.delete(`/pipeline_modelview/api/${pipelineId}`);
};

// 流水线编辑提交
const pipeline_modelview_edit = (pipelineId: number | string, data: IPipelineEdit): Promise<any> => {
  return ajax.put({
    url: `/pipeline_modelview/api/${pipelineId}`,
    data,
  });
};

// 运行流水线
const pipeline_modelview_run = (pipelineId: number | string): Promise<any> => {
  return ajax.post({
    url: `/pipeline_modelview/api/run_pipeline/${pipelineId}`,
  });
};

// 克隆流水线
const pipeline_modelview_copy = (pipelineId: number | string): Promise<any> => {
  return ajax.post({
    url: `/pipeline_modelview/api/copy_pipeline/${pipelineId}`,
  });
};

// 往流水线中添加task
const task_modelview_add = (pipelineId: number | string, data: ITaskAdd): Promise<any> => {
  return ajax.post({
    url: '/task_modelview/api/',
    data: {
      ...data,
      filters: [
        {
          col: 'pipeline',
          opr: 'rel_o_m',
          value: +pipelineId,
        },
      ],
    },
  });
};

// 获取流水线中相应的task
const task_modelview_get = (taskId: string | number): Promise<any> => {
  return ajax.get(`/task_modelview/api/${taskId}`);
};

// 删除对应的 task
const task_modelview_del = (taskId: string | number): Promise<any> => {
  return ajax.delete(`/task_modelview/api/${taskId}`);
};

// 编辑 task
const task_modelview_edit = (pipelineId: string | number, taskId: string | number, data: ITaskEdit): Promise<any> => {
  return ajax.put({
    url: `/task_modelview/api/${taskId}`,
    data: {
      ...data,
      filters: [
        {
          col: 'pipeline',
          opr: 'rel_o_m',
          value: +pipelineId,
        },
      ],
    },
  });
};

const api = {
  getTemplateCommandConfig,
  pipeline_modelview_save,
  job_template_modelview,
  project_all,
  project_modelview,
  pipeline_modelview_add,
  pipeline_modelview_demo,
  pipeline_modelview_list,
  pipeline_modelview_all,
  pipeline_modelview_detail,
  pipeline_modelview_delete,
  pipeline_modelview_edit,
  pipeline_modelview_run,
  pipeline_modelview_copy,
  task_modelview_add,
  task_modelview_get,
  task_modelview_del,
  task_modelview_edit,
};

export default api;

import numpy
import pandas,csv,json,os,sys,datetime,time
import pysnooper

# @pysnooper.snoop()
def draw_line(df):
    xs = df.iloc[:, 0].to_list() # 读取第一例为x
    legends = df.columns.tolist()[1:]  # 首行为header
    data = df.iloc[:, 1:]   # 后续的列为多条折线数据

    series=[]
    for index,legend in enumerate(legends):
        serie = data.iloc[:, index].to_list()
        serie_str = {
            "name": legend,
            "type": 'line',
            "smooth": True,
            "stack": 'Total',
            "data": serie
        }
        series.append(serie_str)

    options = {
      "tooltip": {
        "trigger": 'axis'
      },
      "legend": {
        "data": legends
      },
      "toolbox": {
        "feature": {
          "saveAsImage": {}
        }
      },
      "xAxis": {
        "type": 'category',
        "boundaryGap": False,
        "data": xs
      },
      "yAxis": {
        "type": 'value'
      },
      "series": series
    }
    options = json.dumps(options,indent=4,ensure_ascii=False)
    print(options)
    return options


def draw_bar(df):
    columns = df.columns.tolist()
    # 将列名和数据合并为一个二维数组
    data = numpy.vstack([columns, df.values]).tolist()
    print(data)
    options = {
      "legend": {},
      "tooltip": {},
      "dataset": {
        "source": data
      },
      "xAxis": { "type": 'category' },
      "yAxis": {},
      "series": [{ "type": 'bar' }, { "type": 'bar' }, { "type": 'bar' }]
    }
    options = json.dumps(options, indent=4, ensure_ascii=False)
    print(options)
    return options

def draw_pie(df):
    data = df.iloc[:, 1:]
    # print(data)
    legends = df.columns.tolist()[1:]  # 首行为header
    xs = df.iloc[:, 0].to_list()  # 读取第一例为x
    series=[]
    titles=[]
    for index,legend in enumerate(legends):
        serie = data.iloc[:, index].to_list()
        serie_data=[]
        for x_index,x in enumerate(xs):
            serie_data.append({
                "name":x,
                "value":serie[x_index]
            })
        left = int((index+1)*100/(len(legends)+1))
        radius = 100//(len(legends)+1)
        series.append({
            "type": 'pie',
            "radius": f"{radius}%",
            "center": [f'{left}%','50%'],
            "datasetIndex": index+1,
            "label": {
                "position": 'outer'
            },
            "data": serie_data,
        })
        titles.append({
            "subtext": legend,
            "left": f'{left}%',
            "top": '80%',
            "textAlign": 'center'
        })

    options = {
        "tooltip": {
            "trigger": 'item'
        },
        "legend": {
            "orient": 'horizontal'
        },
        "title":titles,
        "series": series
    }
    
    options = json.dumps(options, indent=4, ensure_ascii=False)
    print(options)
    return options


def draw_scatter(df):
    x = df.iloc[:, 0].to_list()  # 读取第一例为x
    legends = df.columns.tolist()[1:]  # 首行为header
    data = df.iloc[:, 1:]  # 后续的列为多条折线数据

    series = []
    for index, legend in enumerate(legends):
        serie = data.iloc[:, index].to_list()
        serie_str = {
            "name": legend,
            "type": 'scatter',
            "smooth": True,
            "stack": 'Total',
            "data": serie
        }
        series.append(serie_str)

    options = {
        "tooltip": {
            "trigger": 'axis'
        },
        "legend": {
            "data": legends
        },
        "toolbox": {
            "feature": {
                "saveAsImage": {}
            }
        },
        "xAxis": {
            "type": 'category',
            "boundaryGap": False,
            "data": x
        },
        "yAxis": {
            "type": 'value'
        },
        "series": series
    }
    options = json.dumps(options, indent=4, ensure_ascii=False)
    print(options)
    return options

def draw_radar(df):
    xs = df.iloc[:, 0].to_list()  # 读取第一例为x
    legends = df.columns.tolist()[1:]  # 首行为header
    data = df.iloc[:, 1:]  # 后续的列为多条折线数据

    series = []

    for index, legend in enumerate(legends):
        serie = data.iloc[:, index].to_list()
        serie_str = {
            "name": legend,
            "value": serie
        }
        series.append(serie_str)

    # 获取每个x的最大值
    indicator = []
    for index, x in enumerate(xs):
        max_value = max(data.iloc[index].to_list())
        indicator.append({
            "name":x,
            "max":max_value
        })


    options = {
        "tooltip": {
            "trigger": 'axis'
        },
        "legend": {
            "data": legends
        },
        "toolbox": {
            "feature": {
                "saveAsImage": {}
            }
        },
        "radar": {
            "shape": 'circle',
            "indicator": indicator

        },
        "series": [
            {
                "type": 'radar',
                "data":series
            }
        ]
    }
    options = json.dumps(options, indent=4, ensure_ascii=False)
    print(options)
    return options

# 绘制k线图，第一列是x，后面4列是k线，后面的列是折线
# @pysnooper.snoop()
def draw_candlestick(df):
    xs = df.iloc[:, 0].to_list()  # 读取第一例为x
    ys = df.iloc[:, 1:]
    legends = df.columns.tolist()[1:]  # 首行为header

    if len(legends)<4:  # 至少要4列
        return ''

    k_series=df.iloc[:, 1:5].values.tolist()
    line_series =[]
    if len(legends)>4:
        for index,legend in enumerate(legends):
            if index<4:
                continue
            line_series.append(
                {
                    "name": legend,
                    "type": 'line',
                    "data": ys.iloc[:, index].to_list(),
                    "smooth": True,
                    "lineStyle": {
                        "opacity": 0.5
                    }
                }
            )


    options = {
        "tooltip": {
            "trigger": 'axis'
        },
        "legend": {
            "data": legends
        },
        "toolbox": {
            "feature": {
                "saveAsImage": {}
            }
        },
        "xAxis": {
            "data": xs
        },
        "yAxis": {},
        "series": [
            {
                "type": 'candlestick',
                "data": k_series
            }
        ]+line_series
    }
    options = json.dumps(options, indent=4, ensure_ascii=False)
    print(options)
    return options


def draw_heatmap(df):
    xs = df.iloc[:, 0].to_list()  # 读取第一例为x
    ys = df.iloc[:, 1:]
    legends = df.columns.tolist()[1:]  # 首行为header

    data = pandas.melt(df.reset_index(), id_vars=['index'], value_vars=df.columns)
    data = data.values.tolist()
    # 获取所有元素的最大值和最小值
    max_value = float(ys.max().max())
    min_value = float(ys.min().min())
    # print(max_value,min_value)

    options = {
        "tooltip": {
            "trigger": 'axis'
        },
        "legend": {
            "data": legends
        },
        "toolbox": {
            "feature": {
                "saveAsImage": {}
            }
        },
        "xAxis": {
            "type": 'category',
            "data": xs,
            "splitArea": {
              "show": True
            }
          },
        "yAxis": {
            "type": 'category',
            "data": legends,
            "splitArea": {
              "show": True
            }
          },
        "visualMap": {
            "calculable": True,
            "orient": 'horizontal',
            "left": 'center',
            "min":min_value,
            "max":max_value
        },
        "series": [
            {
                "type": 'heatmap',
                "data": data,
                "label": {
                    "show": True
                }
            }
        ]
    }
    options = json.dumps(options, indent=4, ensure_ascii=False)
    print(options)
    return options

def draw_tree(data):

    # 如果不是一个根，就聚合成一个根
    if type(data)==list:
        data = {
            "name": 'root',
            "children": data
        }

    options = {
        "tooltip": {
            "trigger": 'axis'
        },
        "toolbox": {
            "feature": {
                "saveAsImage": {}
            }
        },
        "series": [
            {
                "type": 'tree',
                "top": '10%',
                "left": '10%',
                "bottom": '10%',
                "right": '10%',
                "edgeShape": 'polyline',
                "initialTreeDepth": 10,
                "label": {
                    "backgroundColor": '#fff',
                    "position": 'left',
                    "verticalAlign": 'middle',
                    "align": 'right'
                },
                "leaves": {
                    "label": {
                        "position": 'right',
                        "verticalAlign": 'middle',
                        "align": 'left'
                    }
                },
                "expandAndCollapse": True,
                "data": [data]
            }
        ]
    }
    options = json.dumps(options, indent=4, ensure_ascii=False)
    print(options)


def draw_sunburst(data):
    options = {
        "series": {
            "type": 'sunburst',
            "data": data,
            "radius": [0, '90%'],
            "label": {
                "rotate": 'radial'
            }
        }
    }
    options = json.dumps(options, indent=4, ensure_ascii=False)
    print(options)
    return options

def draw_funnel(df):
    xs = df.iloc[:, 0].to_list()  # 读取第一例为x
    legends = df.columns.tolist()[1:]  # 首行为header
    data = df.iloc[:, 1:]  # 后续的列为多条折线数据
    series=[]
    for index,legend in enumerate(legends):
        serie = data.iloc[:, index].to_list()
        opacity = 1-index/len(legends)
        serie_data = []
        for x_index, x in enumerate(xs):
            serie_data.append({
                "name": x,
                "value": serie[x_index]
            })

        serie_str = {
            "name": legend,
            "type": 'funnel',
            "labelLine": {
                "show": False
            },
            "itemStyle": {
                "opacity": opacity
            },
            "data": serie_data
        }
        series.append(serie_str)


    options = {
        "tooltip": {
            "trigger": 'item',
            "formatter": '{a} <br/>{b} : {c}%'
        },
        "legend": {
            "data": xs
        },
        "toolbox": {
            "feature": {
                "saveAsImage": {}
            }
        },
        "series": series
    }
    options = json.dumps(options, indent=4, ensure_ascii=False)
    print(options)
    return options

# 绘制平行坐标系
# @pysnooper.snoop()
def draw_parallel(df):
    columns_type=df.dtypes.tolist()
    columns=df.columns.tolist()
    parallelAxis=[]
    visualMap_min=None
    visualMap_max=None
    for index,column in enumerate(columns):
        schema={
            "dim":index,
            "name":column
        }
        column_type = columns_type[index]
        # 对于字符串枚举类型，设置类目
        from numpy import dtype
        if column_type==dtype('O'):
            schema['type']='category'
        else:
            # 设置最大最小值
            one_column_data = df[column].to_list()
            min_value,max_value = min(one_column_data),max(one_column_data)
            schema['min'] = min_value
            schema['max'] = max_value
            visualMap_min = min_value
            visualMap_max = max_value
        parallelAxis.append(schema)

    options = {
        "parallelAxis": parallelAxis,
        "tooltip": {
            "padding": 10,
            "backgroundColor": '#222',
            "borderColor": '#777',
            "borderWidth": 1
        },
        "visualMap": {
            "show": True,
            "min":visualMap_min,
            "max":visualMap_max,
            "color": ['#d94e5d', '#eac736', '#50a3ba']
        },
        "series": {
            "type": 'parallel',
            "lineStyle": {
                "width": 2
            },
            "data": df.values.tolist()  # 完成的把数据集取出来
        }
    }
    options = json.dumps(options,indent=4,ensure_ascii=False)
    print(options)
    return options

# @pysnooper.snoop()
def draw(chart_type,file_path):
    try:
        if '.csv' in file_path:
            example = pandas.read_csv(file_path, header=0)
        elif '.json' in file_path:
            example = json.load(open(file_path))
        else:
            return ''
        if chart_type=='line':
            return draw_line(df=example)
        if chart_type=='bar':
            return draw_bar(df=example)
        if chart_type=='pie':
            return draw_pie(df=example)
        if chart_type=='scatter':
            return draw_scatter(df=example)
        if chart_type=='radar':
            return draw_radar(df=example)
        if chart_type=='candlestick':
            return draw_candlestick(df=example)
        if chart_type=='heatmap':
            return draw_heatmap(df=example)
        if chart_type=='tree':
            return draw_tree(data=example)
        if chart_type=='sunburst':
            return draw_sunburst(data=example)
        if chart_type=='parallel':
            return draw_parallel(df=example)
    except Exception as e:
        print(e)
if __name__=="__main__":
    example = pandas.read_csv('data.csv', header=0)
    # draw_line(df=example)
    # draw_bar(df=example)
    # draw_pie(df=example)
    # draw_scatter(df=example)
    # draw_radar(df=example)
    # draw_candlestick(df=example)
    # draw_heatmap(df=example)
    # draw_funnel(df=example)
    draw_parallel(df=example)

    # example = json.load(open('data.json'))
    # draw_tree(data=example)
    # draw_sunburst(data=example)

# import json
# metric = {
#     "metric_type":"echart-parallel",
#     "file_path":"/mnt/admin/pipeline/ray_hyperparams/hyperparams_r_lightgbm.csv"
# }
# json.dump(metric,open('/metric.json',mode='w'))
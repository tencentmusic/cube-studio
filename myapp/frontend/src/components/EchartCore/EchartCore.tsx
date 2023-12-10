import React, { useEffect, useState } from 'react'
import * as echarts from 'echarts';
// import * as echarts from 'echarts/core';
// import {
//     BarChart,
//     // 系列类型的定义后缀都为 SeriesOption
//     BarSeriesOption,
//     PieChart,
//     PieSeriesOption,
//     LineChart,
//     LineSeriesOption,
//     HeatmapChart,
//     HeatmapSeriesOption
// } from 'echarts/charts';
// import {
//     TitleComponent,
//     // 组件类型的定义后缀都为 ComponentOption
//     TitleComponentOption,
//     TooltipComponent,
//     TooltipComponentOption,
//     GridComponent,
//     GridComponentOption,
//     // 数据集组件
//     DatasetComponent,
//     DatasetComponentOption,
//     LegendComponent,
//     // 内置数据转换器组件 (filter, sort)
//     TransformComponent,
//     CalendarComponentOption,
//     CalendarComponent,
//     VisualMapComponent,
//     VisualMapComponentOption,
//     ToolboxComponent
// } from 'echarts/components';
import { LabelLayout, UniversalTransition } from 'echarts/features';
import { CanvasRenderer } from 'echarts/renderers';
import './EchartCore.less';
import { Spin } from 'antd';
import { FieldNumberOutlined } from '@ant-design/icons';
import { useTranslation } from 'react-i18next';

export type ECOption = echarts.EChartsOption
// 通过 ComposeOption 来组合出一个只有必须组件和图表的 Option 类型
// export type ECOption = echarts.ComposeOption<
//     | BarSeriesOption
//     | LineSeriesOption
//     | TitleComponentOption
//     | TooltipComponentOption
//     | GridComponentOption
//     | DatasetComponentOption
//     | CalendarComponentOption
//     | HeatmapSeriesOption
//     | VisualMapComponentOption
//     | PieSeriesOption
// >;

// // 注册必须的组件
// echarts.use([
//     LegendComponent,
//     TitleComponent,
//     TooltipComponent,
//     GridComponent,
//     DatasetComponent,
//     TransformComponent,
//     CalendarComponent,
//     VisualMapComponent,
//     ToolboxComponent,
//     BarChart,
//     LineChart,
//     PieChart,
//     LabelLayout,
//     HeatmapChart,
//     UniversalTransition,
//     CanvasRenderer
// ]);

interface IProps {
    // option: ECOption
    option: echarts.EChartsOption
    loading?: boolean
    title?: string
    style?: React.CSSProperties
    unit?: string
    data?: {
        xData: any[]
        yData: any[]
    }
    isNoData?: boolean
}

const defaultChartStyle: React.CSSProperties = {
    height: 300
}

// https://echarts.apache.org/handbook/zh/how-to/data/dynamic-data
export default function EchartCore(props: IProps) {
    const [chartInstance, setChartInstance] = useState<echarts.ECharts>()
    const id = Math.random().toString(36).substring(2);
    const { t, i18n } = useTranslation();

    const option = {}

    useEffect(() => {
        const chartDom = document.getElementById(id)
        if (chartDom) {
            const chart = echarts.init(chartDom);
            chart.setOption({ ...option, ...props.option })

            if (!chartInstance) {
                setChartInstance(chart)
            }
        }
    }, [props.option, props.data])

    return (
        <Spin spinning={props.loading}>
            <div className="chart-container">
                <div id={id} style={{ ...defaultChartStyle, ...props.style }}></div>
                {
                    props.isNoData ? <div className="chart-nodata">
                        <div>{t('暂无数据')}</div>
                    </div> : null
                }
            </div>
        </Spin>
    )
}

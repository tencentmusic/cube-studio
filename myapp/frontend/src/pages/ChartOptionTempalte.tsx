import React, { useEffect, useState } from 'react'
import { actionADUGTemplateChartOption } from '../api/kubeflowApi'
import EchartCore, { ECOption } from '../components/EchartCore/EchartCore'

interface IProps {
    url?: string
}

export default function ChartOptionTempalte(props: IProps) {
    const [option, setOption] = useState<ECOption>({})
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        if (props.url) {
            actionADUGTemplateChartOption(`${props.url}echart`, {}).then(res => {
                const option = res.data.result
                var currentOps: any = {}
                eval(`currentOps=${option}`)
                setOption(currentOps)
            }).catch(err => { }).finally(() => {
                setLoading(false)
            })
        }
    }, [props.url])

    return (
        <EchartCore option={option} loading={loading} />
    )
}

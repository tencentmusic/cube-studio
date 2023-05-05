import { AxiosResponse } from 'axios'
import axios, { AxiosResFormat } from '.'
import { ILayoutConfig, INodeDetailItem, INodeItem } from '../pages/CommonPipeline/TreePlusInterface'

export const getNodeRelation = (pipelineId: string): AxiosResFormat<{
    dag: INodeItem[]
}> => {
    return axios.get(`/workflow_modelview/api/web/dag/idc/pipeline/${pipelineId}`)
}

export const getPipelineConfig = (pipelineId: string): AxiosResFormat<Array<Record<string, string>>> => {
    return axios.get(`/workflow_modelview/api/web/layout/idc/pipeline/${pipelineId}`)
}

export const getNodeInfo = (pipelineId: string, nodeId: string): AxiosResFormat<{ detail: INodeDetailItem[] }> => {
    return axios.get(`/workflow_modelview/api/web/node_detail/idc/pipeline/${pipelineId}/${nodeId}`)
}

export const getNodeRelationCommon = (url: string): AxiosResFormat<{
    dag: INodeItem[]
    layout: ILayoutConfig
}> => {
    return axios.get(url)
}

export const getNodeInfoCommon = (url: string): AxiosResFormat<{ detail: INodeDetailItem[] }> => {
    return axios.get(url)
}

export const getNodeInfoApi = (url: string): AxiosResFormat<any> => {
    return axios.get(url)
}

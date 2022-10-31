import { AxiosResponse } from 'axios'
import axios, { AxiosResFormat } from '.'
import { IADUGTemplateInfo, IAppHeaderItem, IAppMenuItem, ICustomDialog } from './interface/kubeflowInterface'
import { IAppInfo, IResultItem } from './interface/stateInterface'

export const getAppInfo = (): Promise<AxiosResponse<IAppInfo>> => {
    return axios.get('/app1/info')
}

export const submitData = (url: string, data: Record<any, any>): Promise<AxiosResponse<{
    result: IResultItem[]
}>> => {
    return axios.post(url, data)
}


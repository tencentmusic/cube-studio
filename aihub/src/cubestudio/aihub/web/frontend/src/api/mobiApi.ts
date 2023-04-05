import { AxiosResponse } from 'axios'
import axios, { AxiosResFormat } from '.'
import { IAppInfo, IResultItem } from './interface/stateInterface'

export const getAppInfo = (url: string): Promise<AxiosResponse<IAppInfo>> => {
    return axios.get(`${url}/info`)
}

export const submitData = (url: string, data: Record<any, any>): Promise<AxiosResponse<{
    result: IResultItem[]
}>> => {
    return axios.post(url, data)
}
import { AxiosResponse } from 'axios'
import axios, { AxiosResFormat } from '.'
import { TTaskStatus } from '../pages/DataSearch/interface'

export const actionRun = (params: {
    sql: string,
    [key: string]: any
}): Promise<AxiosResponse<{
    err_msg: string
    log_url: string
    task_id: string
    task_url: string
}>> => {
    return axios.post('/idex/submit_task', params)
}

export const actionGetDataSearchRes = (task_id: string): Promise<AxiosResponse<{
    err_msg: string
    result: Array<Array<string | number>>
    state: TTaskStatus
    task_url: string
    result_url: string
    stage: any
    spark_log_url: string
    spark_ui_url: string
}>> => {
    return axios.get(`/idex/look/${task_id}`,)
}

export const getIdexDBList = (): Promise<AxiosResponse<{
    dbs: string[]
}>> => {
    return axios.get(`/idex/get_user_db`)
}

export const getIdexTableList = (table: string): Promise<AxiosResponse<{
    tables: string[]
}>> => {
    return axios.get(`/idex/get_user_db_tables/${table}`)
}

export const getIdexTaskDownloadUrl = (id: string, separator: string): Promise<AxiosResponse<{
    download_url: string
}>> => {
    return axios.get(`/idex/download_url/${id}`, {
        params: {
            separator
        }
    })
}

export const getIdexTaskResult = (id: string): Promise<AxiosResponse<{
    err_msg: string
    result: Array<Array<string | number>>
}>> => {
    return axios.get(`/idex/result/${id}`)
}

export const stopIndxTask = (id: string): Promise<AxiosResponse<{
    err_msg: string
    result: Array<Array<string | number>>
}>> => {
    return axios.get(`/idex/stop/${id}`)
}


export const getIndexResourceOverview = (group_id: string): Promise<AxiosResponse<{
    err_msg: string
    result: Array<Array<string | number>>
}>> => {
    return axios.get(`/idex/get_resource/${group_id}`)
}

export interface IIdexFormConfigItem {
    id: string,
    label: string,
    type: 'input' | 'select' | 'input-select',
    value: IIdexFormConfigOption[]
    defaultValue: string
    multiple: boolean
    disable: boolean
    placeHolder: string
}

export interface IIdexFormConfigOption {
    label: string
    value: string
    relate: {
        relateId: string
        value: IIdexFormConfigOption[]
    }
}

export const getIdexFormConfig = (): Promise<AxiosResponse<{
    result: IIdexFormConfigItem[]
}>> => {
    return axios.get(`/idex/config`)
}
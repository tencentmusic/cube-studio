import { AxiosResponse } from 'axios'
import axios, { AxiosResFormat } from '.'

export const actionRun = (params: {
    tdw_app_group: string,
    sql: string
}): Promise<AxiosResponse<{
    err_msg: string
    log_url: string
    task_id: string
    task_url: string
}>> => {
    return axios.post('/idex/submit_task', params)
}

export type TTaskStatus = 'init' | 'running' | 'success' | 'failure'

export const actionGetDataSearchRes = (task_id: string): Promise<AxiosResponse<{
    err_msg: string
    result: Array<Array<string | number>>
    state: TTaskStatus
    task_url: string
    result_url: string
    spark_log_url: string
    spark_ui_url: string
}>> => {
    return axios.get(`/idex/look/${task_id}`,)
}
import { AxiosResponse } from 'axios'
import axios, { AxiosResFormat } from '.'
import { IADUGTemplateInfo, IAppHeaderItem, IAppMenuItem, ICustomDialog } from './interface/kubeflowInterface'

export const getAppMenu = (): Promise<AxiosResponse<IAppMenuItem[]>> => {
    return axios.get('/myapp/menu')
}

export const getAppHeaderConfig = (): Promise<AxiosResponse<IAppHeaderItem[]>> => {
    return axios.get('/myapp/navbar_right')
}

export const userLogout = (): Promise<AxiosResponse<IAppMenuItem[]>> => {
    return axios.get('/logout')
}

export const getADUGTemplateApiInfo = (url?: string, id?: string): Promise<AxiosResponse<IADUGTemplateInfo>> => {
    return axios.get(`${url || ''}_info`, {
        params: {
            id
        }
    })
}

export const getCustomDialog = (url: string, signal: AbortSignal): Promise<AxiosResponse<ICustomDialog>> => {
    return axios.get(`/myapp/feature/check?url=${url}`, { signal })
}

export const getADUGTemplateList = (url?: string, params?: any): AxiosResFormat<any> => {
    return axios.get(url || '', { params })
}

export const getData = (url?: string, params?: any): AxiosResFormat<any> => {
    return axios.get(url || '', { params })
}

export const postData = (url?: string, params?: any): AxiosResFormat<any> => {
    return axios.post(url || '', params)
}

export const putData = (url?: string, params?: {}): AxiosResFormat<any> => {
    return axios.put(url || '', params)
}

export const getADUGTemplateDetail = (url: string, form_data?: any): AxiosResFormat<any> => {
    const formData = form_data || { str_related: 1 };
    return axios.get(`${url}`, {
        params: {
            form_data: JSON.stringify(formData)
        }
    })
}

export const actionADUGTemplateAdd = (url?: string, params?: {}): AxiosResFormat<any> => {
    return axios.post(url || '', params)
}

export const actionADUGTemplateUpdate = (url?: string, params?: {}): AxiosResFormat<any> => {
    return axios.put(url || '', params)
}

export const actionADUGTemplateDelete = (url?: string, params?: {}): AxiosResFormat<any> => {
    return axios.delete(url || '', { params })
}

export const actionADUGTemplateSingle = (url?: string): AxiosResFormat<any> => {
    return axios.get(url || '')
}

export const actionADUGTemplateMuliple = (url?: string, params?: { ids: any[] }): AxiosResFormat<any> => {
    return axios.post(url || '', params)
}

export const actionADUGTemplateDownData = (url: string): AxiosResFormat<any> => {
    return axios.get(url)
}

export const actionADUGTemplateRetryInfo = (url: string, params: any): Promise<AxiosResponse<IADUGTemplateInfo>> => {
    return axios.get(url, { params })
}

export const actionADUGTemplateChartOption = (url?: string, params?: {}): Promise<any> => {
    return axios.get(url || '', { params })
}

export const actionADUGTemplateFavorite = (url?: string, params?: {}): AxiosResFormat<any> => {
    return axios.post(url || '', params)
}

export const actionADUGTemplateCancelFavorite = (url?: string, params?: {}): AxiosResFormat<any> => {
    return axios.delete(url || '', { params })
}
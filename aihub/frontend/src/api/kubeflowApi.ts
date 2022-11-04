import { AxiosResponse } from 'axios'
import axios, { AxiosResFormat } from '.'
import { IAppHeaderItem, IAppMenuItem } from './interface/kubeflowInterface'

export const getAppMenu = (): Promise<AxiosResponse<IAppMenuItem[]>> => {
    return axios.get('/myapp/menu')
}

export const userLogout = (): Promise<AxiosResponse<IAppMenuItem[]>> => {
    return axios.get('/logout')
}
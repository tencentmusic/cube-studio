export interface IDataSearchItem {
    tabId: string
    title: string
    appGroup: string
    status: TTaskStatus
    sqlContent?: string
    sqlContentTemporary?: string
    downloadUrl?: string
    reqId?: string
    timer?: any
}
export interface IDataSearchItemParams {
    tabId?: string
    title?: string
    appGroup?: string
    status?: TTaskStatus
    sqlContent?: string
    sqlContentTemporary?: string
    downloadUrl?: string
    reqId?: string
    timer?: any
}

export type IDataSearchStore = {
    [tabId: string]: IDataSearchItem
}


export type TTaskStatus = 'init' | 'running' | 'success' | 'failure' | 'stop';

export type TTaskStep = 'start' | 'parse' | 'execute' | 'end';

export type IEditorStore = {
    [tabId: string]: IEditorItem
}
export interface IEditorItem {
    tabId: string
    title: string
    status: TTaskStatus
    taskMap: Record<string, IEditorTaskItem>
    content?: string
    smartContent?: string
    smartTimer?: any
    smartShow?: boolean
    loading?: boolean
    smartCache?: string
    [key: string]: any
}

export interface IEditorItemParams {
    tabId?: string
    title?: string
    status?: TTaskStatus
    taskMap?: Record<string, IEditorTaskItem>
    content?: string
    appGroup?: string
    database?: string
    table?: string
    biz?: string
    smartContent?: string
    smartTimer?: any
    smartShow?: boolean
    loading?: boolean
    smartCache?: string
}

export interface IEditorTaskItem {
    reqId: string
    status: TTaskStatus
    step: TTaskStep
    message: string
    name?: string
    content?: string
    startTime?: string
    endTime?: string
    duration?: string
    timer?: any
    log?: string
    result?: Array<Array<string | number>>
    downloadUrl?: string
    database?: string
    table?: string
}

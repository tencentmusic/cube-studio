export type ITNodeDetail = 'common' | 'table' | 'sql'

export interface INodeItem {
    "nid": string,
    "pid": string,
    "title": string,
    "pod": string,
    "start_time": string,
    "finish_time": string,
    "detail_url": string,
    "name": string,
    "outputs": any,
    "icon": string,
    "status": {
        "label": string,
        "icon": string
    },
    "message": string,
    "node_shape": string,
    "color": string,
    "task_name": string,
    "task_id": string,
    "task_label": string,
    "volumeMounts": IVolumeMountItem[],
    "children": INodeItem[]
    parent: INodeItem[]
}

export interface ILayoutConfig {
    "create_time": string,
    "search": string,
    "status": string,
    "cluster": string,
    "pipeline-id": string,
    "pipeline-rtx": string,
    "run-id": string,
    "run-rtx": string,
    "save-time": string,
    "schedule_type": string,
    "workflow-name": string,
    "workflows.argoproj.io/completed": string,
    "workflows.argoproj.io/phase": string,
    "progress": string,
    "start_time": string,
    "finish_time": string,
    "pipeline-name": string,
    "pipeline-describe": string,
    "icon": string,
    "title": string,
    "right_button": Array<{
        "label": string,
        "url": string
    }>
    detail: Array<ILayoutDetailItem[]>
}
export interface ILayoutDetailItem {
    label: string
    name: string
    value: string
}

export interface INodeDetailItem {
    tabName: string,
    content: [{
        groupName: string,
        groupContent: IGroupContentItem
    }],
    bottomButton: IButtonItem[]
}

export interface IButtonItem {
    text: string,
    url: string,
    icon: string
}

export interface IGroupContentItem {
    label: string,
    value: any,
    type: TGroupContentType
}

export type TGroupContentType = 'map' | 'iframe' | 'echart' | 'text' | 'api' | 'html'

export interface ITreeNode {
    nid: string | number;
    pid?: string | number;
    cn_name: string;
    en_name?: string;
    create_time?: string
    update_time?: string
    data_fields: TTreeNodeType;
    status: TTreeNodeStatus;
    parent?: ITreeNode[]
    children?: ITreeNode[]
}
export interface IVolumeMountItem {
    "mountPath": string,
    "name": string,
    "subPath": string
}

export type TTreeNodeType = 'ROOT' | 'BUSINESS' | 'THEME' | 'COLLECT' | 'OTHER'

export type TTreeNodeStatus = 'ENABLE' | 'DISABLE'

export interface ITreeParams {
    nodeId: string
    nodeType: TTreeNodeType
    depthChildren: number
    depthParent: number
}

export interface ITreeNodeDetailItem {
    cn: string
    key: string
    value: string
    type: ITNodeDetail
}
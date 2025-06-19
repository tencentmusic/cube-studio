
export interface ITabDetailItem {
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

export type TGroupContentType = 'map' | 'iframe' | 'echart' | 'text' | 'html'| 'markdown' | 'image'

export interface IActionButton {
    text: string,
    url: string,
    icon?: string,
    arg?: string,
    method: "get" | "post" | "delete"
}

export interface ITabsModalData {
    title: string;
    content:ITabDetailItem[];
    bottomButton:IActionButton[];
}
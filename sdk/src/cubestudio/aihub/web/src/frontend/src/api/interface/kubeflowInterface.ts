export interface IAppMenuItem {
    icon: string,
    name: string,
    title: string,
    isMenu?: boolean,
    menu_type: string,
    model_name?: string,
    url?: string,
    breadcrumbs?: string[],
    hidden?: boolean,
    disable?: boolean
    isSubRoute?: boolean,
    children: IAppMenuItem[],
    related?: IAppMenuItem[],
    isExpand?: boolean
    path?: string
}

export interface IAppHeaderItem {
    text: string
    icon?: string
    link: string
    pic_url?: string
}

export interface IColumnRelateItem {
    des_columns: string[];
    src_columns: string[];
    related: {
        src_value: string[];
        des_value: string[];
    }[];
}

export interface IADUGTemplateActionItem {
    confirmation: string
    icon: string
    multiple: boolean
    name: string
    single: boolean
    text: string
}

export interface ICustomDialog {
    content: string
    delay: number
    target: string
    title: string
    type: string
    hit: boolean
}


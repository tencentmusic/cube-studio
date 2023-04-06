import { IColumnConfigItem } from "../../pages/columnConfig";

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
export interface IADUGTemplateInfo {
    action?: Record<string, IADUGTemplateActionItem>
    add_columns: Array<Record<string, any>>
    add_title: string
    description_columns: Record<string, any>
    edit_columns: Array<Record<string, any>>
    edit_title: string
    filters: Record<string, {
        filter: Array<{ name: string, operator: string }>
        default: string
        'ui-type': string
        values: Array<{ id: number, value: string }>
    }>
    label_columns: Record<string, string>
    list_columns: string[]
    list_title: string
    order_columns: string[]
    permissions: string[]
    show_columns: string[]
    show_title: string
    help_url: string | null
    route_base: string
    add_fieldsets: Array<{
        expanded: boolean
        group: string
        fields: string[]
    }>
    edit_fieldsets: Array<{
        expanded: boolean
        group: string
        fields: string[]
    }>
    primary_key: string
    label_title: string
    column_related: Record<string, IColumnRelateItem>
    cols_width: Record<string, IColumnConfigItem>
    import_data: boolean
    download_data: boolean
    list_ui_type?: 'card' | 'table'
    list_ui_args?: {
        card_width: string
        card_height: string
    }
    ops_link: Array<{
        text: string
        url: string
    }>
    enable_favorite: boolean
    echart: boolean
    page_size: number
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
    style: Record<string, any>
}


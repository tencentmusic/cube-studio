import { RouteObject } from "react-router-dom";

export interface IRouterConfigPlusItem {
    title?: string
    name?: string
    icon?: any
    isMenu?: boolean,
    disable?: boolean,
    menu_type?: string,
    url?: string,
    state?: any,
    baseApi?: string
    isSubRoute?: boolean
    hidden?: boolean
    children?: any[]
    related?: any[]
    appIndex?: number
    key?: string
    isLocalMenu?: boolean
    isCollapsed?: boolean
    isCloseSide?: boolean
    isExpand?: boolean
}




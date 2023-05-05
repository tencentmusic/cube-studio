import { IndexRouteObject, NonIndexRouteObject, RouteObject } from "react-router-dom";

export interface IRouterConfigPlusItem extends Omit<RouteObject, 'children' | 'index'> {
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
    children?: IRouterConfigPlusItem[]
    related?: IRouterConfigPlusItem[]
    appIndex?: number
    key?: string
    isLocalMenu?: boolean
    isCollapsed?: boolean
    isCloseSide?: boolean
    isExpand?: boolean
    index?: boolean
    isSingleModule?: boolean
    isDropdown?: boolean
}




import { DashboardOutlined, DatabaseOutlined, ExperimentOutlined, HomeOutlined, InboxOutlined, ProfileOutlined, SearchOutlined, SettingOutlined, ToolOutlined, UserOutlined } from '@ant-design/icons';
import React from 'react';
import { IRouterConfigPlusItem } from './api/interface/baseInterface';
import { IAppMenuItem } from './api/interface/kubeflowInterface';
import Page404 from './pages/Page404';
import Cookies from 'js-cookie'
import LoadingStar from './components/LoadingStar/LoadingStar';
const userName = Cookies.get('myapp_username')
const isAdmin = userName === 'admin'

const LoadingComponent = () => {
    return <div className="d-f ac jc w100 h100">
        <LoadingStar />
    </div>
}

const lazy2Compont = (factory: () => Promise<{
    default: () => JSX.Element;
}>, props?: any) => {
    const Component = React.lazy(() => factory())
    const id = Math.random().toString(36).substring(2)
    return <React.Suspense key={id} fallback={<LoadingComponent />}>
        <Component {...props} />
    </React.Suspense>
}

export const innerDynamicRouterConfig: any[] = [
    // {
    //     path: '/xxx',
    //     title: 'xxx',
    //     key: 'xxx',
    //     icon: '',
    //     menu_type: 'innerRouter',
    //     isCollapsed: true,
    //     element: lazy2Compont(() => import("./pages/xxx"))
    // },
]

const innerDynamicRouterConfigMap = innerDynamicRouterConfig.reduce((pre, next) => ({
    ...pre,
    [next.key || '']: next
}), {}) as Record<string, IRouterConfigPlusItem>;

export const routerConfigPlus: any[] = [
    {
        path: '/',
        index: true,
        element: lazy2Compont(() => import("./pages/Index/Index"))
    },
    ...innerDynamicRouterConfig,
    { path: '*', element: <Page404 /> },
]

export const createRoute = (urlName: string) => {
    return {
        path: `/${urlName}`,
        element: lazy2Compont(() => import("./pages/Index/Index"), {
            isCollapsed: true,
            isSubRoute: true,
        })
    }
}

export const getDefaultOpenKeys = (data: any[]) => {
    const openKeys: string[] = []
    const quene = [...data]
    while (quene.length) {
        const item = quene.shift()
        if (item?.isExpand && item.path) {
            openKeys.push(item.path)
        }
        if (item?.children) {
            quene.push(...item.children)
        }
    }
    return openKeys
}
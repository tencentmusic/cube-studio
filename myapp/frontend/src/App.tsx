import React, { useEffect, useState } from 'react';
import {
  BrowserRouter as Routers,
  Route,
  Link,
  useRoutes,
  useNavigate,
  BrowserRouterProps,
  useLocation
} from "react-router-dom";
import { IRouterConfigPlusItem } from './api/interface/baseInterface';
import { Dropdown, Menu, Select, Spin } from 'antd';
import { formatRoute, getDefaultOpenKeys, routerConfigPlus } from './routerConfig';
import SubMenu from 'antd/lib/menu/SubMenu';
import { clearWaterNow, drawWater, drawWaterNow, getParam, obj2UrlParam, parseParam2Obj } from './util'
import { getAppHeaderConfig, getAppMenu } from './api/kubeflowApi';
import { GithubOutlined, LeftOutlined, QuestionCircleOutlined, RightOutlined } from '@ant-design/icons';
import Cookies from 'js-cookie'
import { changeTheme } from './theme';
import { handleTips } from './api';
import { IAppHeaderItem } from './api/interface/kubeflowInterface';

const RouterConfig = (config: any[]) => {
  let element = useRoutes(config);
  return element;
}

const getAppList = (config: IRouterConfigPlusItem[]) => config.filter(item => !!item.name && !item.hidden).map((item, index) => ({ ...item, appIndex: index }))

interface IProps { }

const AppWrapper = (props: IProps) => {
  const [currentRoute, setcurrentRoute] = useState('/')
  const [openKeys, setOpenKeys] = useState<string[]>([])
  const [currentAppIndex, setCurrentAppIndex] = useState<number>()
  const [currnetRouteConfig, setCurrnetRouteConfig] = useState(routerConfigPlus)
  const [currentAppList, setCurrentAppList] = useState(getAppList(routerConfigPlus))
  const [CurrentRouteComponent, setCurrentRouteComponent] = useState<any>()
  const [isMenuCollapsed, setIsMenuCollapsed] = useState(false)
  const [imgUrlProtraits, setImgUrlProtraits] = useState('')
  const [headerConfig, setHeaderConfig] = useState<IAppHeaderItem[]>([])

  const navigate = useNavigate();
  const location = useLocation()

  useEffect(() => {
    const { pathname } = location
    let tarPath = '/'
    if (pathname === '/') {
      clearWaterNow()
    } else {
      tarPath = pathname
      drawWaterNow()
    }
    handleChangePageTitle(pathname, currnetRouteConfig)

    const currentOpenKeys = tarPath.split('/').slice(0, 3).join('/')
    setcurrentRoute(tarPath)
    setOpenKeys([...openKeys, currentOpenKeys])
    handleAppIndex(currnetRouteConfig.filter(item => !!item.name && !item.hidden))
  }, [location])

  useEffect(() => {
    getAppMenu().then(res => {
      const dynamicRoute = formatRoute(res.data).map(item => ({ ...item }))
      const tarRoute = [...routerConfigPlus, ...dynamicRoute]
      const appList = getAppList(tarRoute)
      const { pathname } = location
      const defaultOpenKeys = getDefaultOpenKeys(tarRoute)

      handleChangePageTitle(pathname, tarRoute)

      setOpenKeys(defaultOpenKeys)

      setCurrnetRouteConfig(tarRoute)
      setCurrentRouteComponent(() => () => RouterConfig(tarRoute))
      setCurrentAppList(appList)
      handleAppIndex(appList)
    }).catch(err => {
      const tarRoute = [...routerConfigPlus]
      const appList = getAppList(tarRoute)
      const { pathname } = location
      const defaultOpenKeys = getDefaultOpenKeys(tarRoute)

      handleChangePageTitle(pathname, tarRoute)

      setOpenKeys(defaultOpenKeys)

      setCurrnetRouteConfig(tarRoute)
      setCurrentRouteComponent(() => () => RouterConfig(tarRoute))
      setCurrentAppList(appList)
      handleAppIndex(appList)
    })

    getAppHeaderConfig().then(res => {
      const config = res.data
      setHeaderConfig(config)
    }).catch(err => { })
  }, [])

  const handleAppIndex = (appList: IRouterConfigPlusItem[]) => {
    const { pathname } = location
    const appPath = `/${pathname.split('/')[1]}`
    let appIndex = undefined
    for (let idx = 0; idx < appList.length; idx++) {
      const app = appList[idx];
      if (app.path === appPath) {
        appIndex = idx
      }
    }
    setCurrentAppIndex(appIndex)
  }

  const handleClickApp = (app: IRouterConfigPlusItem, index: number) => {
    if (app.path === '/') {
      commitUrlChange('/')
      setCurrentAppIndex(index)
      navigate(app.path || '/')
    } else if (app.menu_type === 'iframe' && app.path) {
      commitUrlChange(app.path)
      setcurrentRoute(app.path)
      setCurrentAppIndex(index)
      navigate(app.path)
    } else if (app.menu_type === 'out_link' && app.path) {
      window.open(app.url, 'blank')
    } else {
      let currentItem = app
      while (currentItem && currentItem.children) {
        currentItem = currentItem.children[0]
      }
      if (currentItem) {
        let appMenuPath = currentItem.path || ''
        commitUrlChange(appMenuPath)
        setcurrentRoute(appMenuPath)
        setCurrentAppIndex(index)
        navigate(appMenuPath)
      }
    }

  }

  const handleChangePageTitle = (pathname: string, currnetRouteConfig: IRouterConfigPlusItem[]) => {
    const currentAppName = '/' + pathname.substring(1).split('/')[0] || ''
    const routerMap: Record<string, IRouterConfigPlusItem> = currnetRouteConfig.reduce((pre: any, next) => ({ ...pre, [`${next.path || ''}`]: next }), {})
    const currentRoute = routerMap[currentAppName]
    if (currentRoute && currentRoute.title) {
      document.title = `cube - ${currentRoute.title}`
    } else {
      document.title = 'cube计算平台'
    }
  }

  const commitUrlChange = (key: string) => {
    if (window !== window.top) {
      const locationTop = (window as any).top.location
      const href = locationTop.href
      const path = locationTop.origin + locationTop.pathname
      const search = href.split('?').slice(1).join('?')
      const paramObj = parseParam2Obj(search)
      paramObj['pathUrl'] = key;
      const paramStr = obj2UrlParam(paramObj);
      const currentUrl = path + '#/?' + paramStr;

      (window as any).top.location.href = currentUrl
    }
  }

  const renderMenu = (currentAppIndex: number | undefined) => {

    if (currentAppIndex === undefined || !currentAppList.length || !currentAppList[currentAppIndex] || !currentAppList[currentAppIndex].children?.length) {
      return null
    }
    const currentAppMenu = currentAppList[currentAppIndex].children


    if (currentAppMenu && currentAppMenu.length) {

      const menuContent = currentAppList[currentAppIndex].children?.map(menu => {
        if (menu.isMenu) {
          return <SubMenu key={menu.path} title={menu.title}>
            {
              menu.children?.map(sub => {
                if (sub.isMenu) {
                  return <Menu.ItemGroup key={sub.path} title={sub.title}>
                    {
                      sub.children?.map(thr => {
                        return <Menu.Item disabled={!!thr.disable} hidden={!!thr.hidden} key={thr.path} onClick={() => {
                          if (!menu.isCollapsed) {
                            setIsMenuCollapsed(false)
                          }
                          if (thr.menu_type === 'out_link' || thr.menu_type === 'in_link') {
                            window.open(thr.url, 'blank')
                          } else {
                            navigate(thr.path || '')
                          }
                        }}>
                          <div className="icon-wrapper">
                            {
                              Object.prototype.toString.call(thr.icon) === '[object String]' ? <div className="icon-custom svg16 mr8" dangerouslySetInnerHTML={{ __html: thr.icon }}></div> : sub.icon
                            }
                            {/* <div className="icon-custom svg16 mr8" dangerouslySetInnerHTML={{ __html: thr.icon }}></div> */}
                            {thr.title}
                          </div>
                        </Menu.Item>
                      })
                    }
                  </Menu.ItemGroup>
                }
                return <Menu.Item disabled={!!sub.disable} hidden={!!sub.hidden} key={sub.path} onClick={() => {
                  if (!menu.isCollapsed) {
                    setIsMenuCollapsed(false)
                  }
                  if (sub.menu_type === 'out_link' || sub.menu_type === 'in_link') {
                    window.open(sub.url, 'blank')
                  } else {
                    navigate(sub.path || '')
                  }
                }}>
                  <div className="icon-wrapper">
                    {
                      Object.prototype.toString.call(sub.icon) === '[object String]' ? <div className="icon-custom svg16 mr8" dangerouslySetInnerHTML={{ __html: sub.icon }}></div> : sub.icon
                    }
                    {sub.title}
                  </div>
                </Menu.Item>
              })
            }
          </SubMenu>
        }
        return <Menu.Item disabled={!!menu.disable} hidden={!!menu.hidden} key={menu.path} onClick={() => {
          if (!menu.isCollapsed) {
            setIsMenuCollapsed(false)
          }
          if (menu.menu_type === 'out_link' || menu.menu_type === 'in_link') {
            window.open(menu.url, 'blank')
          } else {
            navigate(menu.path || '')
          }
        }}>
          <div className="icon-wrapper">
            {
              Object.prototype.toString.call(menu.icon) === '[object String]' ? <div className="icon-custom svg16 mr8" dangerouslySetInnerHTML={{ __html: menu.icon }}></div> : menu.icon
            }
            {menu.title}
          </div>
        </Menu.Item>
      })

      return <div className="side-menu">
        <div className="h100 ov-h" style={{ width: isMenuCollapsed ? 0 : 'auto' }}>
          <Menu
            selectedKeys={[currentRoute]}
            openKeys={openKeys}
            mode="inline"
            onOpenChange={(openKeys) => {
              console.log('openKeys', openKeys)
              setOpenKeys(openKeys)
            }}
            onSelect={(info) => {
              const key = info.key
              commitUrlChange(key)
              setcurrentRoute(key)
              // navigate(key)
            }}
          >
            {menuContent}
          </Menu>
        </div>
        <div className="menu-collapsed cp">
          <span className="w100 h100 d-f jc ac p-a" onClick={() => {
            setIsMenuCollapsed(!isMenuCollapsed)
          }}>{
              isMenuCollapsed ? <RightOutlined style={{ color: '#d9d9d9' }} /> : <LeftOutlined style={{ color: '#d9d9d9' }} />
            }</span>
          <img src={require('./images/sideLeft.png')} alt="" />
        </div>
      </div>
    }
    return null
  }

  return (
    <div className="content-container fade-in">
      {/* Header */}
      <div className="navbar">
        <div className="d-f ac pl24 h100">
          <div className="cp" onClick={() => {
            setCurrentAppIndex(undefined)
            setcurrentRoute('/')
            navigate('/')
          }}>
            <img style={{ height: 42 }} src={require("./images/logoCB.svg").default} alt="img" />
          </div>

          {
            currentAppList.length ? <ul className="mainapp-topmenu">
              {
                currentAppList.map((app) => {
                  const currentAppName = '/' + currentRoute.substring(1).split('/')[0] || ''
                  return <li
                    onClick={() => handleClickApp(app, app.appIndex || 0)}
                    key={`appList${app.appIndex}`}
                    className={[currentAppName === app.path ? 'mainapp-item-active' : ''].join(' ')}
                  // style={
                  //   currentAppName === app.path ? { color: '#1e1653', borderTop: '2px solid #1e1653' } : {}
                  // }
                  >
                    {
                      Object.prototype.toString.call(app.icon) === '[object String]' ? <div className="icon-custom" dangerouslySetInnerHTML={{ __html: app.icon }}></div> : app.icon
                    }
                    <div className="mainapp-topmenu-name">{app.title}</div>
                  </li>
                })
              }
            </ul> : null
          }
        </div>



        <div className="d-f ac plr16 h100">

          {/* <div className="mr16">
            <Select
              bordered={false}
              defaultValue="star"
              options={[
                { label: '无垠星空主题', value: 'star' },
                { label: '清新蓝调主题', value: 'blue' },
                // { label: '披星戴月', value: 'dark' },
              ]}
              onChange={(value: any) => {
                changeTheme(value)
              }}
            />
          </div> */}

          {
            headerConfig.map(config => {
              if (config.icon) {
                return <a
                  href={config.link}
                  target="_blank"
                  className="mr12 d-f ac"
                >
                  <span className="pr4">{config.text}</span><span className="icon-custom" dangerouslySetInnerHTML={{ __html: config.icon }}></span>
                </a>
              } else if (config.pic_url) {
                return <a
                  href={config.link}
                  target="_blank"
                  className="mr12 d-f ac"
                >
                  <span className="pr4">{config.text}</span><img style={{ height: 30 }} src={config.pic_url} alt="" />
                </a>
              }
            })
          }

          <Dropdown overlay={<Menu>
            <Menu.Item onClick={() => {
              setCurrentAppIndex(undefined)
              navigate('/user')
            }}>用户中心</Menu.Item>
            <Menu.Item onClick={() => {
              Cookies.remove('myapp_username');
              handleTips.userlogout()
            }}>退出登录</Menu.Item>
          </Menu>
          }>
            <img className="mr8 cp" style={{ borderRadius: 200, height: 32 }} src={imgUrlProtraits} onError={() => {
              setImgUrlProtraits(require('./images/male.png'))
            }} alt="img" />
          </Dropdown>
        </div>
      </div>

      <div className="main-content-container">

        {renderMenu(currentAppIndex)}

        <div className="ov-a w100 bg-title">
          {
            CurrentRouteComponent && <CurrentRouteComponent />
          }
        </div>

      </div >
    </div>
  );
};

export default AppWrapper;

import React, { useEffect, useState } from 'react';
import {
  BrowserRouter as Routers,
  useRoutes,
  useNavigate,
  useLocation,
  RouteObject
} from "react-router-dom";

import { Drawer, Dropdown, Menu, Select, Spin, Tag } from 'antd';
import { IRouterConfigPlusItem } from './api/interface/baseInterface';
import { formatRoute, getDefaultOpenKeys, routerConfigPlus } from './routerConfig';
import SubMenu from 'antd/lib/menu/SubMenu';
import { clearWaterNow, drawWater, drawWaterNow, getParam, obj2UrlParam, parseParam2Obj } from './util'
import { getAppHeaderConfig, getAppMenu, getCustomDialog, userLogout } from './api/kubeflowApi';
import { IAppHeaderItem, IAppMenuItem, ICustomDialog } from './api/interface/kubeflowInterface';
import { AppstoreOutlined, DownOutlined, LeftOutlined, RightOutlined, TranslationOutlined } from '@ant-design/icons';
import Cookies from 'js-cookie'
import { handleTips } from './api';
import globalConfig from './global.config'
import i18nEn from './images/i18nEn.svg';

import { useTranslation, Trans, } from 'react-i18next';
const userName = Cookies.get('myapp_username')

const RouterConfig = (config: RouteObject[]) => {
  let element = useRoutes(config);
  return element;
}

const getRouterMap = (routerList: IRouterConfigPlusItem[]): Record<string, IRouterConfigPlusItem> => {
  const res: Record<string, IRouterConfigPlusItem> = {}
  const queue = [...routerList]
  while (queue.length) {
    const item = queue.shift()
    if (item) {
      res[item?.path || ''] = item
      if (item?.children && item.children.length) {
        queue.push(...item.children)
      }
    }
  }
  return res
}

const getValidAppList = (config: IRouterConfigPlusItem[]) => config.filter(item => !!item.name && !item.hidden)

interface IProps { }

const AppWrapper = (props: IProps) => {
  const [openKeys, setOpenKeys] = useState<string[]>([])
  const [currentNavList, setCurrentNavList] = useState<IRouterConfigPlusItem[]>([])
  const [sourceAppList, setSourceAppList] = useState<IRouterConfigPlusItem[]>([])
  const [sourceAppMap, setSourceAppMap] = useState<Record<string, IRouterConfigPlusItem>>({})
  const [CurrentRouteComponent, setCurrentRouteComponent] = useState<any>()
  const [isMenuCollapsed, setIsMenuCollapsed] = useState(false)
  const [isShowSlideMenu, setIsShowSlideMenu] = useState(true)
  const [imgUrlProtraits, setImgUrlProtraits] = useState('')
  const [customDialogVisable, setCustomDialogVisable] = useState(false)
  const [customDialogInfo, setCustomDialogInfo] = useState<ICustomDialog>()
  const [headerConfig, setHeaderConfig] = useState<IAppHeaderItem[]>([])
  const [navSelected, setNavSelected] = useState<string[]>([])
  const isShowNav = getParam('isShowNav')

  const navigate = useNavigate();
  const location = useLocation()

  const { t, i18n } = useTranslation();

  useEffect(() => {
    getAppMenu().then(res => {
      const remoteRoute = res.data
      const dynamicRoute = formatRoute([...remoteRoute])
      const tarRoute = [...dynamicRoute, ...routerConfigPlus]
      const tarRouteMap = getRouterMap(tarRoute)

      setSourceAppList(tarRoute)
      setSourceAppMap(tarRouteMap)

      const defaultOpenKeys = getDefaultOpenKeys(tarRoute)
      setOpenKeys(defaultOpenKeys)

      setCurrentRouteComponent(() => () => RouterConfig(tarRoute as RouteObject[]))
    }).catch(err => { })

    getAppHeaderConfig().then(res => {
      const config = res.data
      setHeaderConfig(config)
    }).catch(err => { })
  }, [])

  useEffect(() => {
    if (sourceAppList.length && Object.keys(sourceAppMap).length) {
      const { pathname } = location
      if (pathname === '/') {
        clearWaterNow()
      } else {
        drawWaterNow()
      }
      handleCurrentRoute(sourceAppMap, getValidAppList(sourceAppList))
    }
  }, [location, sourceAppList, sourceAppMap])

  useEffect(() => {
    const controller = new AbortController()
    const url = encodeURIComponent(location.pathname)
    getCustomDialog(url, controller.signal).then(res => {
      setCustomDialogInfo(res.data)
      setCustomDialogVisable(res.data.hit)
    }).catch(err => {
      console.log(err);
    })
    return () => {
      controller.abort()
    }
  }, [location])

  const handleCurrentRoute = (appMap: Record<string, IRouterConfigPlusItem>, appList: IRouterConfigPlusItem[]) => {
    const { pathname } = location
    const [_, stLevel, edLevel] = pathname.split('/')
    const stLevelApp = appMap[`/${stLevel}`]
    let currentNavKey = ""
    if (stLevelApp && stLevelApp.isSingleModule) {
      currentNavKey = `/${stLevel}/${edLevel}`
    } else {
      currentNavKey = `/${stLevel}`
    }

    let topNavAppList = appList
    if (stLevelApp && stLevelApp.isSingleModule) {
      topNavAppList = stLevelApp.children || []
    }

    setCurrentNavList(topNavAppList)
    setNavSelected([currentNavKey])
    setIsShowSlideMenu(stLevelApp && !stLevelApp.isCollapsed)
  }

  const handleClickNav = (app: IRouterConfigPlusItem, subPath?: string) => {
    if (app.path === '/') {
      navigate(app.path || '/')
    } else if (app.menu_type === 'iframe' && app.path) {
      navigate(app.path)
    } else if (app.menu_type === 'out_link' && app.url) {
      window.open(app.url, 'blank')
    } else if (app.menu_type === 'in_link' && app.path) {
      window.open(app.url, 'blank')
    } else {
      const currentApp = sourceAppMap[subPath || '']
      let currentItem = subPath ? currentApp : app

      while (currentItem && currentItem.children) {
        currentItem = currentItem.children[0]
      }

      if (currentItem) {
        let appMenuPath = currentItem.path || ''
        navigate(appMenuPath)
      }
    }
  }

  const renderMenu = () => {
    const { pathname } = location
    const currentNavMap = sourceAppMap
    const [currentSelected] = navSelected

    if (currentNavMap && currentSelected && currentNavMap[currentSelected]?.children?.length) {

      const currentAppMenu = currentNavMap[currentSelected].children
      if (currentAppMenu && currentAppMenu.length) {

        const menuContent = currentAppMenu.map(menu => {
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
          <div className="h100 ov-h d-f fd-c" style={{ width: isMenuCollapsed ? 0 : 'auto' }}>
            <Menu
              selectedKeys={[pathname]}
              openKeys={openKeys}
              mode="inline"
              onOpenChange={(openKeys) => {
                setOpenKeys(openKeys)
              }}
              onSelect={(info) => {
                const key = info.key
              }}
            >
              {menuContent}
            </Menu>
            <div className="p16 ta-r bor-t" style={{ borderColor: '#e5e6eb' }}>
              <div className="d-il bor-l pl16" style={isMenuCollapsed ? { position: 'absolute', bottom: 16, left: 0, borderColor: '#e5e6eb' } : { borderColor: '#e5e6eb' }}>
                {
                  isMenuCollapsed ? <RightOutlined className="cp" onClick={() => {
                    setIsMenuCollapsed(!isMenuCollapsed)
                  }} /> : <LeftOutlined className="cp" onClick={() => {
                    setIsMenuCollapsed(!isMenuCollapsed)
                  }} />
                }
              </div>
            </div>
          </div>
        </div>
      }
    }

    return null
  }

  const renderNavTopMenu = () => {
    return currentNavList.map((app) => {
      if (!!app.hidden) {
        return null
      }
      if (app.isSingleModule || app.isDropdown) {
        return <Menu.SubMenu key={app.path} title={
          <div className="star-topnav-submenu" onClick={() => {
            if (app.isDropdown) {
              return
            }
            handleClickNav(app)
          }}>
            {
              Object.prototype.toString.call(app.icon) === '[object String]' ? <div className="icon-custom" dangerouslySetInnerHTML={{ __html: app.icon }}></div> : app.icon
            }
            <div className="mainapp-topmenu-name">{app.title}</div>
            <DownOutlined className="ml8" />
          </div>
        }>
          {
            (app.children || []).map(subapp => {
              return <Menu.Item key={subapp.path} onClick={() => {
                handleClickNav(subapp, subapp.path)
              }}>
                <div className="d-f ac">
                  {
                    Object.prototype.toString.call(subapp.icon) === '[object String]' ? <div className="icon-custom" dangerouslySetInnerHTML={{ __html: subapp.icon }}></div> : subapp.icon
                  }
                  <div className="pl8">{subapp.title}</div>
                </div>
              </Menu.Item>
            })
          }
        </Menu.SubMenu>
      }
      return <Menu.Item key={app.path} onClick={() => {
        handleClickNav(app)
      }}>
        {
          Object.prototype.toString.call(app.icon) === '[object String]' ? <div className="icon-custom" dangerouslySetInnerHTML={{ __html: app.icon }}></div> : app.icon
        }
        <div className="mainapp-topmenu-name">{app.title}</div>
      </Menu.Item>
    })
  }

  const renderSingleModule = () => {
    const { pathname } = location
    const [_, stLevel] = pathname.split('/')
    const stLevelApp = sourceAppMap[`/${stLevel}`]
    if (stLevelApp && stLevelApp.isSingleModule) {
      return <Tag color="#1672fa">{stLevelApp.title}</Tag>
    }
    return null
  }

  return (
    <div className="content-container fade-in">
      {/* Header */}
      {
        isShowNav === 'false' ? null : <div className="navbar">
          <div className="d-f ac pl48 h100">
            <div className="d-f ac">
              <div className="cp pr16" style={{ width: 'auto' }} onClick={() => {
                navigate('/', { replace: true })
              }}>
                <img style={{ height: 42 }} src={globalConfig.appLogo.default} alt="img" />
              </div>

              {
                renderSingleModule()
              }
            </div>
            <div className="star-topmenu">
              <Menu mode="horizontal" selectedKeys={navSelected}>
                {renderNavTopMenu()}
              </Menu>
            </div>
          </div>

          <div className="d-f ac plr16 h100">
            <div>
              {/* <img className='c-theme' src={require('./images/i18nEn.svg').default} alt="" /> */}
              {
                i18n.language === 'zh-CN' ? <div onClick={() => {
                  i18n.changeLanguage('en')
                  window.location.reload()
                }} className='icon-custom c-theme cp mr8' dangerouslySetInnerHTML={{
                  __html: `
                  <svg t="1699066332987" class="icon" viewBox="0 0 1070 1024" version="1.1"
                      xmlns="http://www.w3.org/2000/svg" p-id="1494"
                      xmlns:xlink="http://www.w3.org/1999/xlink" width="66.875" height="64">
                      <path d="M232.582 358.133c12.288 36.33 32.59 67.851 60.905 95.633 24.042-26.18 42.207-58.235 53.96-95.633H232.583z" p-id="1495"></path>
                      <path d="M981.615 143.36H532.836L507.192 6.055H90.468C44.522 6.055 7.123 43.453 7.123 89.4v708.43c0 45.946 37.399 83.344 83.345 83.344h379.86l-30.453 137.305h541.74c45.947 0 83.345-37.398 83.345-83.344v-708.43c0-45.947-37.398-83.345-83.345-83.345zM415.833 564.358c-49.152-18.165-89.756-41.139-122.346-67.852-34.192 30.453-76.933 52.892-126.62 66.783l-17.096-28.316c48.618-12.822 89.222-32.055 121.277-59.303-33.124-33.658-56.097-72.66-68.92-117.003h-46.48v-32.056h121.277c-7.48-13.89-17.096-27.247-28.316-40.07l32.056-11.753c11.22 14.425 21.37 31.522 30.453 51.29h115.935v32.055h-46.481c-14.96 45.946-36.33 84.413-64.646 115.4 31.522 25.11 71.057 45.947 117.538 63.043l-17.631 27.782zM1023.288 934.6c0 22.974-18.7 41.673-41.673 41.673H492.232l20.837-95.633h156.538l-89.222-497.397-0.534 2.671-3.74-19.767 1.069 0.534-32.59-181.649h437.56c22.973 0 41.672 18.7 41.672 41.673V934.6z" p-id="1496"></path>
                      <path d="M684.566 541.384h114.866v-30.453H684.566v-60.905h122.346v-30.453H648.771V638.62h162.95v-30.453H684.565v-66.783zM924.45 475.67c-9.616 0-18.164 1.603-26.178 5.877-7.48 3.74-14.96 9.617-20.837 17.096v-18.699h-34.727V638.62h34.727v-95.633c1.069-12.822 5.343-22.439 12.823-29.384 6.41-5.877 13.89-9.083 22.439-9.083 24.041 0 35.795 12.823 35.795 39.001v94.565h34.727v-97.77c1.069-43.275-19.233-64.646-58.769-64.646z" p-id="1497"></path>
                  </svg>
                ` }}></div> : <div onClick={() => {
                  i18n.changeLanguage('zh-CN')
                  window.location.reload()
                }} className='icon-custom c-theme cp mr8' dangerouslySetInnerHTML={{
                  __html: `
                  <svg t="1699066345106" class="icon" viewBox="0 0 1024 1024" version="1.1"
                      xmlns="http://www.w3.org/2000/svg" p-id="1656"
                      xmlns:xlink="http://www.w3.org/1999/xlink" width="64" height="64">
                      <path d="M580 477.7h-9.8l37.6 209.1c24.8-8.8 47.4-22.2 67.5-39.6-20.6-24.8-37.1-52.5-50-81.9l39.6-5.2c10.8 22.2 23.2 42.2 37.6 59.3 29.3-35.5 51.5-82.9 67.5-142.2l-190 0.5z m149.9 169.5c22.7 19.6 48.4 34 77.2 42.7l18.1 5.7-10.8 38.6-18.1-5.7c-34.5-10.8-66.5-28.8-93.7-53.1-25.3 22.7-55.1 40.2-87.5 51l25.3 141.1H489.8l-20.1 92.2h472.4c22.2 0 40.2-18.1 40.2-40.2v-683c0-22.2-18.1-40.2-40.2-40.2H520.3l31.4 175.2-1-0.5 3.6 19.1 0.5-2.6 8.8 50H661v-40.2h75.3v40.2H862v40.2h-52.5c-17.7 70.6-44.6 127.3-79.6 169.5z m-281.3 220H82.3C38 867.2 2 831.1 2 786.8V104.2c0-44.8 36-80.3 80.3-80.3h401.8l24.8 132.4h432.8c44.3 0 80.3 36 80.3 80.3v683.1c0 44.3-36 80.3-80.3 80.3H419.8l28.8-132.8zM259.1 558.1v-42.2h-79.3v-62.4h73.7v-41.7h-73.7v-53.1h79.3V317H133.3v241.1h125.8z m193.6 0V437.5c0-21.7-5.2-38.6-15-50.5s-24.8-17.5-44.3-17.5c-11.4 0-21.7 2.1-30.4 6.7s-16 11.9-20.6 20.6h-2.6l-6.2-23.7h-35V558h45.3v-87c0-21.7 3.1-37.1 8.8-46.9 5.7-9.3 15-13.9 27.8-13.9 9.3 0 16 3.1 20.6 9.8 4.1 6.7 6.7 16.5 6.7 29.8V558l44.9 0.1z" p-id="1657"></path>
                  </svg>
                ` }}></div>
              }
            </div>
            {
              headerConfig.map(config => {
                if (config.icon) {
                  return <a
                    href={config.link}
                    target="_blank"
                    className="mr12 d-f ac" rel="noreferrer"
                  >
                    <span className="pr4">{config.text}</span><span className="icon-custom" dangerouslySetInnerHTML={{ __html: config.icon }}></span>
                  </a>
                } else if (config.pic_url) {
                  return <a
                    href={config.link}
                    target="_blank"
                    className="mr12 d-f ac" rel="noreferrer"
                  >
                    <span className="pr4">{config.text}</span><img style={{ height: 30 }} src={config.pic_url} alt="" />
                  </a>
                }
              })
            }

            <Dropdown overlay={<Menu>
              <Menu.Item onClick={() => {
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
      }

      <div className="main-content-container">
        {isShowSlideMenu ? renderMenu() : null}

        <div className="ov-a w100 bg-title p-r" id="componentContainer">
          {/* 自定义弹窗 */}
          {
            customDialogVisable ? <Drawer
              getContainer={false}
              style={{ position: 'absolute', height: 'calc(100vh - 100px)', top: '10%', ...customDialogInfo?.style }}
              bodyStyle={{ padding: 0 }}
              mask={false}
              contentWrapperStyle={{ width: 'auto' }}
              title={customDialogInfo?.title} placement="right" onClose={() => { setCustomDialogVisable(false) }}
              visible={customDialogVisable}>
              <div className="h100" dangerouslySetInnerHTML={{ __html: customDialogInfo?.content || '' }}></div>
            </Drawer> : null
          }
          {
            CurrentRouteComponent && <CurrentRouteComponent />
          }
        </div>

        {
          customDialogInfo?.content ? <div className="c-text-w fs12 p-f" style={{ backgroundColor: 'transparent', zIndex: 10, right: 16, bottom: 32 }}>
            <div className="bg-theme d-f jc ac cp" style={{ borderRadius: 6, width: 36, height: 36 }} onClick={() => {
              setCustomDialogVisable(true)
            }}><AppstoreOutlined style={{ color: '#fff', fontSize: 22 }} /></div>
          </div> : null
        }

      </div >
    </div>
  );
};

export default AppWrapper;

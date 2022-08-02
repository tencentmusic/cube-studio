import React from 'react';
import ReactDOM from 'react-dom';
import App from './App';
import reportWebVitals from './reportWebVitals';
import './index.less';
import zhCN from 'antd/lib/locale/zh_CN';
import { ConfigProvider, Spin } from 'antd';
import './store/index'
import cookies from 'js-cookie';

import {
  BrowserRouter, HashRouter
} from "react-router-dom";
import { handleTips } from './api';
import Aegis from '@tencent/aegis-web-sdk';
import { changeTheme } from './theme';
import LoadingStar from './components/LoadingStar/LoadingStar';

// Spin.setDefaultIndicator(<img style={{ width: 64, height: 32, left: 'calc(50% - 21px)' }} src={require('./images/loadingTme.gif')} />)
Spin.setDefaultIndicator(<LoadingStar />)

let isLogin = false
const userName = cookies.get('myapp_username')

if (!!userName) {
  isLogin = true

  if (process.env.NODE_ENV !== 'development') {
    // 如果使用 cdn 的话，Aegis 会自动绑定在 window 对象上
    const aegis = new Aegis({
      id: 'NUehmRjuAcfkWgtOTM', // 项目key
      uin: userName, // 用户唯一 ID（可选）
      reportApiSpeed: true, // 接口测速
      reportAssetSpeed: true, // 静态资源测速
      spa: true, // spa 页面需要开启，页面切换的时候上报pv
    })
  }
} else {
  handleTips.gotoLogin()
}

changeTheme('star')

ReactDOM.render(
  isLogin ?
    <ConfigProvider locale={zhCN}>
      {
        process.env.REACT_APP_ROUTER_TYPE === 'browser' ?
          <BrowserRouter basename={process.env.REACT_APP_BASE_ROUTER || '/'}>
            <App />
          </BrowserRouter> :
          <HashRouter basename={process.env.REACT_APP_BASE_ROUTER || '/'}>
            <App />
          </HashRouter>
      }
    </ConfigProvider> : <></>,
  document.getElementById('root')
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();

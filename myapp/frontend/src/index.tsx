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
import { changeTheme } from './theme';
import LoadingStar from './components/LoadingStar/LoadingStar';

// Spin.setDefaultIndicator(<img style={{ width: 64, height: 32, left: 'calc(50% - 21px)' }} src={require('./images/loadingTme.gif')} />)
Spin.setDefaultIndicator(<LoadingStar />)

let isLogin = false
const userName = cookies.get('myapp_username')

if (!!userName) {
  isLogin = true
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

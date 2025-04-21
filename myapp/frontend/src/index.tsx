import React from 'react';
import ReactDOM from 'react-dom';
import App from './App';
import reportWebVitals from './reportWebVitals';
import './index.less';
import zhCN from 'antd/lib/locale/zh_CN';
import en from 'antd/lib/locale/en_US';
import { ConfigProvider, Spin } from 'antd';
import { getI18n } from 'react-i18next';
import './store/index'
import './locales/i18n'

import {
  BrowserRouter, HashRouter
} from "react-router-dom";
import cookies from 'js-cookie';
import { handleTips } from './api';
import { setTheme } from './theme';
import LoadingStar from './components/LoadingStar/LoadingStar';
import globalConfig from './global.config';

Spin.setDefaultIndicator(<LoadingStar />)

setTheme(globalConfig.theme)

let isLogin = false
const userName = cookies.get('myapp_username')

if (!!userName) {
  isLogin = true
} else {
  handleTips.gotoLogin()
}

ReactDOM.render(
  isLogin ?
    <ConfigProvider locale={getI18n().language === 'zh-CN' ? zhCN : en}>
      <BrowserRouter basename={process.env.REACT_APP_BASE_ROUTER || '/'}>
        <App />
      </BrowserRouter>
    </ConfigProvider> : <></>,
  document.getElementById('root')
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();

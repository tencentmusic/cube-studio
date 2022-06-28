import React from 'react';
import ReactDOM from 'react-dom';
import { HashRouter as Router } from 'react-router-dom';
import { mergeStyles, initializeIcons } from '@fluentui/react';
import AppRouter from './routes';
import { store } from './models/store';
import { Provider } from 'react-redux';

// fluentui icon 资源初始化
initializeIcons();

// 全局样式
// 样式可替换
mergeStyles({
  ':global(body,html,#app)': {
    margin: 0,
    padding: 0,
    height: '100vh',
  },
  ':global(.react-flow__edge-path)': {
    strokeWidth: '2 !important',
  },
});

ReactDOM.render(
  <React.StrictMode>
    <Provider store={store}>
      <Router>
        <AppRouter />
      </Router>
    </Provider>
  </React.StrictMode>,
  document.getElementById('app'),
);

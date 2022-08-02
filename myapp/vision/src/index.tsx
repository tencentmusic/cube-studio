import React from 'react';
import ReactDOM from 'react-dom';
import { HashRouter as Router } from 'react-router-dom';
import { mergeStyles, initializeIcons, createTheme, ThemeProvider } from '@fluentui/react';
import AppRouter from './routes';
import { store } from './models/store';
import { Provider } from 'react-redux';
import './app.less';

// fluentui icon 资源初始化
initializeIcons();

const myTheme = createTheme({
  palette: {
    themePrimary: '#1890ff',
    themeLighterAlt: '#f6fbff',
    themeLighter: '#daedff',
    themeLight: '#b9ddff',
    themeTertiary: '#74bcff',
    themeSecondary: '#339cff',
    themeDarkAlt: '#1581e6',
    themeDark: '#116dc2',
    themeDarker: '#0d508f',
    neutralLighterAlt: '#faf9f8',
    neutralLighter: '#f3f2f1',
    neutralLight: '#edebe9',
    neutralQuaternaryAlt: '#e1dfdd',
    neutralQuaternary: '#d0d0d0',
    neutralTertiaryAlt: '#c8c6c4',
    neutralTertiary: '#595959',
    neutralSecondary: '#373737',
    neutralPrimaryAlt: '#2f2f2f',
    neutralPrimary: '#000000',
    neutralDark: '#151515',
    black: '#0b0b0b',
    white: '#ffffff',
  }
});
// 全局样式
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
  <ThemeProvider style={{ height: '100vh' }} applyTo='body' theme={myTheme}>
    <React.StrictMode>
      <Provider store={store}>
        <Router>
          <AppRouter />
        </Router>
      </Provider>
    </React.StrictMode>
  </ThemeProvider>
  ,
  document.getElementById('app'),
);

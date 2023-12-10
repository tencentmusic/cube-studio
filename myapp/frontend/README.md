# 运行环境

- node@16.13.0

# 启动/调试项目

`npm run start`

# 打包上线

`npm run build`

# 主题自定义

### 更改系统全局设置

```js
// 文件路径：myapp/frontend/src/global.config.ts
const globalConfig = {
  appLogo, //系统左上角logo
  loadingLogo, //系统加载logo
  theme: "blue", //主题设置，可自定义
};
```

### 设置全局主题样式

```js
// 文件路径：myapp/frontend/src/theme.ts
const blue: IThemeConfig = {
  "--ant-primary-color": "#1672fa",
  "--ant-primary-color-hover": "#1672fa",
  "--ant-primary-color-active": "#096dd9",
  "--ant-primary-color-outline": "rgba(24, 144, 255, 0.2)",
  "--ant-primary-1": "#e6f7ff",
  "--ant-primary-2": "#bae7ff",
  "--ant-primary-3": "#91d5ff",
  "--ant-primary-4": "#69c0ff",
  "--ant-primary-5": "#1672fa",
  "--ant-primary-6": "#1672fa",
  "--ant-primary-7": "#096dd9",
  "--ant-primary-color-deprecated-pure": "",
  "--ant-primary-color-deprecated-l-35": "#cbe6ff",
  "--ant-primary-color-deprecated-l-20": "#7ec1ff",
  "--ant-primary-color-deprecated-t-20": "#46a6ff",
  "--ant-primary-color-deprecated-t-50": "#8cc8ff",
  "--ant-primary-color-deprecated-f-12": "rgba(24, 144, 255, 0.12)",
  "--ant-primary-color-active-deprecated-f-30": "rgba(230, 247, 255, 0.3)",
  "--ant-primary-color-active-deprecated-d-02": "#dcf4ff",
  "--ant-success-color": "#52c41a",
  "--ant-success-color-hover": "#73d13d",
  "--ant-success-color-active": "#389e0d",
  "--ant-success-color-outline": "rgba(82, 196, 26, 0.2)",
  "--ant-success-color-deprecated-bg": "#f6ffed",
  "--ant-success-color-deprecated-border": "#b7eb8f",
  "--ant-error-color": "#ff4d4f",
  "--ant-error-color-hover": "#ff7875",
  "--ant-error-color-active": "#d9363e",
  "--ant-error-color-outline": "rgba(255, 77, 79, 0.2)",
  "--ant-error-color-deprecated-bg": "#fff2f0",
  "--ant-error-color-deprecated-border": "#ffccc7",
  "--ant-warning-color": "#faad14",
  "--ant-warning-color-hover": "#ffc53d",
  "--ant-warning-color-active": "#d48806",
  "--ant-warning-color-outline": "rgba(250, 173, 20, 0.2)",
  "--ant-warning-color-deprecated-bg": "#fffbe6",
  "--ant-warning-color-deprecated-border": "#ffe58f",
  "--ant-info-color": "#1890ff",
  "--ant-info-color-deprecated-bg": "#e6f7ff",
  "--ant-info-color-deprecated-border": "#91d5ff",
  "--ant-link": "#1672fa",
};
```

### 更改系统 logo

```js
// 文件路径：myapp/frontend/public/index.html
<img src="%PUBLIC_URL%/cubeStudioLogo.png" alt="">
```

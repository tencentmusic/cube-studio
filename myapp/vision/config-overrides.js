/* eslint-disable @typescript-eslint/no-var-requires */
const MonacoWebpackPlugin = require('monaco-editor-webpack-plugin');
const path = require('path');
const paths = require('react-scripts/config/paths');

// 合并项目，修改打包输出的路径
paths.appBuild = path.join(path.dirname(paths.appBuild), '../static/appbuilder/vison');

module.exports = {
  webpack: config => {
    // alias
    config.resolve.alias = {
      ...config.resolve.alias,
      '@src': path.resolve(__dirname, 'src'),
    };

    // plugin
    config.plugins.push(
      new MonacoWebpackPlugin({
        languages: ['json'],
      }),
    );

    return config;
  },
};

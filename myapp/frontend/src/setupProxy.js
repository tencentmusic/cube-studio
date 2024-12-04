const { createProxyMiddleware } = require('http-proxy-middleware');

// https://create-react-app.dev/docs/proxying-api-requests-in-development/
module.exports = function (app) {
    app.use(
        ['/workflow_modelview'],
        createProxyMiddleware({
            target: 'http://localhost',
            changeOrigin: true,
        })
    );

    app.use(
        ['**/api/**', '/myapp', '/login', '/idex', '/users', '/roles','/static/appbuilder'],  //本地调试pipeline和首页的时候，不要添加/static/appbuilder代理
        createProxyMiddleware({
            target: 'http://localhost',
            changeOrigin: true,
        })
    );
};
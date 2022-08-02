const { createProxyMiddleware } = require('http-proxy-middleware');

// https://create-react-app.dev/docs/proxying-api-requests-in-development/
module.exports = function (app) {
    app.use(
        [
            '**/api/**',
            '/myapp',
            '/etl_pipeline_modelview'
        ],
        createProxyMiddleware({
            target: 'http://localhost',
            changeOrigin: true,
        })
    );
};

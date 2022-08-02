import Axios, { AxiosResponse } from 'axios';
import { notification } from 'antd';
import cookies from 'js-cookie';
const baseApi = process.env.REACT_APP_BASE_URL || 'http://localhost/' || 'http://kubeflow.tke.woa.com'
const baseApiAuth = ''

// console.log(process.env, baseApi)

export type AxiosResFormat<T> = Promise<AxiosResponse<ResponseFormat<T>>>;
export interface ResponseFormat<T = any> {
    message: string;
    result: T;
    data: T;
    status: number
}

const codeMessage: Record<number, string> = {
    200: '服务器成功返回请求的数据。',
    201: '新建或修改数据成功。',
    202: '一个请求已经进入后台排队（异步任务）。',
    204: '删除数据成功。',
    400: '发出的请求有错误，服务器没有进行新建或修改数据的操作。',
    401: '用户没有权限（令牌、用户名、密码错误）。',
    403: '用户得到授权，但是访问是被禁止的。',
    404: '发出的请求针对的是不存在的记录，服务器没有进行操作。',
    406: '请求的格式不可得。',
    410: '请求的资源被永久删除，且不会再得到的。',
    422: '当创建一个对象时，发生一个验证错误。',
    500: '服务器发生错误，请检查服务器。',
    502: '网关错误。',
    503: '服务不可用，服务器暂时过载或维护。',
    504: '网关超时。',
};

/** 异常处理程序 */
const errorHandler = (error: { response: Response }): Response => {
    const { response } = error;
    if (response && response.status) {
        const errorText = codeMessage[response.status] || response.statusText;
        const { status, url } = response;

        notification.error({
            message: `请求错误 ${status}: ${url}`,
            description: errorText,
        });
    } else if (!response) {
        notification.error({
            description: '您的网络发生异常，无法连接服务器',
            message: '网络异常',
        });
    }
    return response;
};

const axios = Axios.create({
    timeout: 600000,
    responseType: 'json',
});


class HandleTips {
    private errorQuene: string[] = [];
    private tipsTimer: NodeJS.Timeout | undefined;
    private errorFlag = true;
    private lock = true;
    /**
     *
     * 只要队列还有内容，就合并相同的错误
     * @param {string} err
     * @memberof HandleTips
     */
    trigger(content: string, type?: 'success' | 'info' | 'warning' | 'error') {
        if (this.errorQuene.indexOf(content) === -1) {
            this.errorQuene.push(content);
        }
        if (this.errorFlag) {
            this.errorFlag = false;
            this.tipsTimer = setInterval(() => {
                if (this.errorQuene.length) {
                    const contentMsg = this.errorQuene.shift();
                    notification[type || 'info']({
                        message: type,
                        description: contentMsg,
                    });
                } else {
                    this.tipsTimer && clearInterval(this.tipsTimer);
                    this.errorFlag = true;
                }
            }, 500);
        }
    }

    gotoLogin() {
        if (this.lock) {
            this.lock = false;
            // const authUrl = `${baseApiAuth}/login?redirect=${window.location.href}`;
            const authUrl = `http://${window.location.host}/login/?login_url=${window.location.href}`
            setTimeout(() => {
                window.location.href = authUrl;
            }, 800);
        }
    }
}

export const handleTips = new HandleTips();

// 请求拦截器
axios.interceptors.request.use(
    (config) => {
        if (!!cookies.get('myapp_username')) {
            let logConfig = config
            return logConfig;
        } else {
            handleTips.gotoLogin();
        }
    },
    (error) => {
        return Promise.reject(error);
    }
);

// 响应拦截器
axios.interceptors.response.use(
    (response) => {
        const { data } = response;
        // response.headers['api_flashes'] = '[["error", "test"]]'
        const tipMessage = JSON.parse(response.headers['api_flashes'] || '[]')
        if (tipMessage && Array.isArray(tipMessage) && tipMessage.length) {
            tipMessage.forEach((tip: any) => {
                if (Array.isArray(tip)) {
                    const [type, content] = tip
                    handleTips.trigger(content, type);
                }
            });
        }
        if (data) {
            if (data.error_code === 0 || data.ret === 0) {
                return response;
            } else if (data.error_code && data.message) {
                let errMsg = ''
                if (Object.prototype.toString.call(data.message) === '[object Object]') {
                    errMsg = JSON.stringify(data.message)
                } else {
                    errMsg = `${data.message}`;
                }
                handleTips.trigger(errMsg);
                throw new Error(errMsg);
            } else {
                return response;
            }
        } else {
            return response;
        }
    },
    (error) => {
        // Any status codes that falls outside the range of 2xx cause this function to trigger
        if (error.response) {
            const tipMessage = JSON.parse(error.response.headers['api_flashes'] || '[]')
            console.log('tipMessage', tipMessage)
            if (tipMessage && Array.isArray(tipMessage) && tipMessage.length) {
                tipMessage.forEach((tip: any) => {
                    if (Array.isArray(tip)) {
                        const [type, content] = tip
                        handleTips.trigger(content, type);
                    }
                });
            }

            const { data } = error.response;

            let errMsg = data
            errMsg = `${data ? data.msg || data.message : error.response.status}`;

            if (data && Object.prototype.toString.call(data.msg || data.message) === '[object Object]') {
                errMsg = JSON.stringify(data.msg || data.message)
                console.log('errMsg', errMsg)
            } if (data && Object.prototype.toString.call(data) === '[object String]') {
                errMsg = data
            }

            if (error.response.status === 401) {
                handleTips.trigger('登录超时，需要重新登录');
                handleTips.gotoLogin();
            } else {
                handleTips.trigger(errMsg);
            }
            // throw new Error(`${data ? data.msg : error.response.status}`);
        }
        return Promise.reject(error);
    }
);

export default axios;

import Axios, { AxiosResponse } from 'axios';
import { notification } from 'antd';
import cookies from 'js-cookie';
import { getI18n } from 'react-i18next';
const baseApi = process.env.REACT_APP_BASE_URL || 'http://localhost/'

export type AxiosResFormat<T> = Promise<AxiosResponse<ResponseFormat<T>>>;
export interface ResponseFormat<T = any> {
    message: string;
    result: T;
    data: T;
    status: number
}

console.log(getI18n())

const axios = Axios.create({
    timeout: 600000,
    responseType: 'json',
});


class HandleTips {
    private errorQuene: string[] = [];
    private tipsTimer: NodeJS.Timeout | undefined;
    private errorFlag = true;
    private lock = true;
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
                        description: <div dangerouslySetInnerHTML={{ __html: contentMsg || '' }}></div>,
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
            const authUrl = `//${window.location.host}/login/?login_url=${window.location.href}`
            setTimeout(() => {
                window.location.href = authUrl;
            }, 800);
        }
    }

    userlogout() {
        const logoutUrl = `//${window.location.host}/logout`
        setTimeout(() => {
            window.location.href = logoutUrl;
        }, 800);
    }

}

export const handleTips = new HandleTips();

// 请求拦截器
axios.interceptors.request.use(
    (config) => {
        config.headers.set('language', getI18n().language)
        if (!!cookies.get('myapp_username')) {
            let logConfig = config
            return logConfig;
        } else {
            handleTips.gotoLogin();
            return Promise.reject('');
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
                    if (Object.prototype.toString.call(tip) === '[object Object]') {
                        handleTips.trigger(tip.content, tip.type);
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

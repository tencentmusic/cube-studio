import axios, { AxiosRequestConfig, AxiosResponse, AxiosError } from 'axios';
import cookie from 'cookie';

const { myapp_username, t_uid, km_uid } = cookie.parse(document.cookie);
const Authorization = myapp_username || t_uid || km_uid || '';

axios.defaults.baseURL =
  process.env.NODE_ENV === 'development' ? process.env.REACT_APP_API_HOST : window.location.origin;

axios.defaults.headers = Object.assign(axios.defaults.headers, {
  'Content-Type': 'application/json',
  Authorization,
});

axios.interceptors.response.use(
  (res: AxiosResponse) => res,
  err => Promise.reject(err),
);

interface IAxiosRequest {
  url: string;
  data?: any;
  options?: any;
}

const Ajax = {
  get: function (url: string, options: any = {}): Promise<any> {
    return new Promise((resolve, reject) => {
      const defaultOptions: AxiosRequestConfig = {
        timeout: 10000,
        responseType: 'json',
      };

      Object.assign(defaultOptions, options);
      axios
        .get(url, defaultOptions)
        .then(response => {
          resolve(response.data);
        })
        .catch(err => {
          this._errHandle(err);
          reject(err);
        });
    });
  },
  delete: function (url: string, options: any = {}): Promise<any> {
    return new Promise((resolve, reject) => {
      const defaultOptions: AxiosRequestConfig = {
        timeout: 10000,
        responseType: 'json',
      };

      Object.assign(defaultOptions, options);
      axios
        .delete(url, defaultOptions)
        .then(response => {
          resolve(response.data);
        })
        .catch(err => {
          this._errHandle(err);
          reject(err);
        });
    });
  },
  post: function ({ url, data, options }: IAxiosRequest): Promise<any> {
    return new Promise((resolve, reject) => {
      const defaultOptions: AxiosRequestConfig = {
        timeout: 10000,
        responseType: 'json',
      };

      Object.assign(defaultOptions, options);
      axios
        .post(url, data, defaultOptions)
        .then(response => {
          resolve(response.data);
        })
        .catch(err => {
          this._errHandle(err);
          reject(err);
        });
    });
  },
  put: function ({ url, data, options }: IAxiosRequest): Promise<any> {
    return new Promise((resolve, reject) => {
      const defaultOptions: AxiosRequestConfig = {
        timeout: 10000,
        responseType: 'json',
      };

      Object.assign(defaultOptions, options);
      axios
        .put(url, data, defaultOptions)
        .then(response => {
          resolve(response.data);
        })
        .catch(err => {
          this._errHandle(err);
          reject(err);
        });
    });
  },
  _errHandle: function (error: AxiosError): void {
    if (error.response) {
      console.error(error.response);
    } else if (error.request) {
      console.error(error.request);
    } else {
      console.error('Error', error.message);
    }
    console.error(error.config);
  },
};

export default Ajax;

import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import LanguageDetector from 'i18next-browser-languagedetector';
// import HttpApi from 'i18next-http-backend';

import mainZh from './main.zh'
import mainEn from './main.en'

const resources = {
    en: {
        translation: {
            ...mainEn
        }
    },
    zh: {
        translation: {
            ...mainZh
        }
    }
}

i18n
    // load translation using http -> see /public/locales
    // learn more: https://github.com/i18next/i18next-http-backend
    // .use(HttpApi)
    // 检测用户当前使用的语言
    // 文档: https://github.com/i18next/i18next-browser-languageDetector
    .use(LanguageDetector)
    // 注入 react-i18next 实例
    .use(initReactI18next)
    // 初始化 i18next
    // 配置参数的文档: https://www.i18next.com/overview/configuration-options
    .init({
        debug: true,
        fallbackLng: 'en',
        // interpolation: {
        //     escapeValue: false,
        // },
        resources
    });

export default i18n;
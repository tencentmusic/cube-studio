interface IThemeConfig {
    [key: string]: string
}

const star: IThemeConfig = {
    '--ant-primary-color': '#1E1653',
    '--ant-primary-color-hover': '#1E1653',
    '--ant-primary-color-active': '#096dd9',
    '--ant-primary-color-outline': 'rgba(24, 144, 255, 0.2)',
    '--ant-primary-1': '#1E16531a',
    '--ant-primary-2': '#bae7ff',
    '--ant-primary-3': '#91d5ff',
    '--ant-primary-4': '#69c0ff',
    '--ant-primary-5': '#1E1653',
    '--ant-primary-6': '#1E1653',
    '--ant-primary-7': '#096dd9',
    '--ant-primary-color-deprecated-pure': '',
    '--ant-primary-color-deprecated-l-35': '#cbe6ff',
    '--ant-primary-color-deprecated-l-20': '#7ec1ff',
    '--ant-primary-color-deprecated-t-20': '#46a6ff',
    '--ant-primary-color-deprecated-t-50': '#8cc8ff',
    '--ant-primary-color-deprecated-f-12': 'rgba(24, 144, 255, 0.12)',
    '--ant-primary-color-active-deprecated-f-30': 'rgba(230, 247, 255, 0.3)',
    '--ant-primary-color-active-deprecated-d-02': '#dcf4ff',
    '--ant-success-color': '#52c41a',
    '--ant-success-color-hover': '#73d13d',
    '--ant-success-color-active': '#389e0d',
    '--ant-success-color-outline': 'rgba(82, 196, 26, 0.2)',
    '--ant-success-color-deprecated-bg': '#f6ffed',
    '--ant-success-color-deprecated-border': '#b7eb8f',
    '--ant-error-color': '#ff4d4f',
    '--ant-error-color-hover': '#ff7875',
    '--ant-error-color-active': '#d9363e',
    '--ant-error-color-outline': 'rgba(255, 77, 79, 0.2)',
    '--ant-error-color-deprecated-bg': '#fff2f0',
    '--ant-error-color-deprecated-border': '#ffccc7',
    '--ant-warning-color': '#faad14',
    '--ant-warning-color-hover': '#ffc53d',
    '--ant-warning-color-active': '#d48806',
    '--ant-warning-color-outline': 'rgba(250, 173, 20, 0.2)',
    '--ant-warning-color-deprecated-bg': '#fffbe6',
    '--ant-warning-color-deprecated-border': '#ffe58f',
    '--ant-info-color': '#1890ff',
    '--ant-info-color-deprecated-bg': '#e6f7ff',
    '--ant-info-color-deprecated-border': '#91d5ff',
    '--ant-link': '#8e264f',
};

const blue: IThemeConfig = {
    '--ant-primary-color': '#1672fa',
    '--ant-primary-color-hover': '#1672fa',
    '--ant-primary-color-active': '#096dd9',
    '--ant-primary-color-outline': 'rgba(24, 144, 255, 0.2)',
    '--ant-primary-1': '#e6f7ff',
    '--ant-primary-2': '#bae7ff',
    '--ant-primary-3': '#91d5ff',
    '--ant-primary-4': '#69c0ff',
    '--ant-primary-5': '#1672fa',
    '--ant-primary-6': '#1672fa',
    '--ant-primary-7': '#096dd9',
    '--ant-primary-color-deprecated-pure': '',
    '--ant-primary-color-deprecated-l-35': '#cbe6ff',
    '--ant-primary-color-deprecated-l-20': '#7ec1ff',
    '--ant-primary-color-deprecated-t-20': '#46a6ff',
    '--ant-primary-color-deprecated-t-50': '#8cc8ff',
    '--ant-primary-color-deprecated-f-12': 'rgba(24, 144, 255, 0.12)',
    '--ant-primary-color-active-deprecated-f-30': 'rgba(230, 247, 255, 0.3)',
    '--ant-primary-color-active-deprecated-d-02': '#dcf4ff',
    '--ant-success-color': '#52c41a',
    '--ant-success-color-hover': '#73d13d',
    '--ant-success-color-active': '#389e0d',
    '--ant-success-color-outline': 'rgba(82, 196, 26, 0.2)',
    '--ant-success-color-deprecated-bg': '#f6ffed',
    '--ant-success-color-deprecated-border': '#b7eb8f',
    '--ant-error-color': '#ff4d4f',
    '--ant-error-color-hover': '#ff7875',
    '--ant-error-color-active': '#d9363e',
    '--ant-error-color-outline': 'rgba(255, 77, 79, 0.2)',
    '--ant-error-color-deprecated-bg': '#fff2f0',
    '--ant-error-color-deprecated-border': '#ffccc7',
    '--ant-warning-color': '#faad14',
    '--ant-warning-color-hover': '#ffc53d',
    '--ant-warning-color-active': '#d48806',
    '--ant-warning-color-outline': 'rgba(250, 173, 20, 0.2)',
    '--ant-warning-color-deprecated-bg': '#fffbe6',
    '--ant-warning-color-deprecated-border': '#ffe58f',
    '--ant-info-color': '#1890ff',
    '--ant-info-color-deprecated-bg': '#e6f7ff',
    '--ant-info-color-deprecated-border': '#91d5ff',
    '--ant-link': '#1672fa',
};

const dark: IThemeConfig = {
    '--ant-primary-color': 'darkgray',
};

const themesCollection: Record<TThemeType, IThemeConfig> = { star, dark, blue };

export type TThemeType = 'dark' | 'blue' | 'star'

export const setTheme = (theme: TThemeType) => {
    const nextTheme = themesCollection[theme];

    Object.keys(nextTheme).forEach((key) => {
        document.documentElement.style.setProperty(key, nextTheme[key]);
    });
};
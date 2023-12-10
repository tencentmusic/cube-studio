import { TThemeType } from "./theme"

const appLogo = require('./images/logoCB.svg')
const loadingLogo = require('./images/logoCB.svg')

interface IGlobalConfig {
    appLogo: any,
    loadingLogo: any,
    theme: TThemeType,
}

const globalConfig: IGlobalConfig = {
    appLogo,
    loadingLogo,
    theme: 'star',
}

export default globalConfig
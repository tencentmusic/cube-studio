import * as React from 'react';
import { configure } from 'mobx';
import CURDMainTemplateStore from "../pages/CURDMainTemplate/CURDMainTemplateStore";

configure({ enforceActions: 'always' })

export const stores = { CURDMainTemplateStore }
export const CounterContext = React.createContext(stores)

export const useStores = () => React.useContext(CounterContext)

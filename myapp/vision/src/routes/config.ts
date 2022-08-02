import React, { lazy } from 'react';

export interface IRoute {
  name: string;
  path: string;
  component: React.LazyExoticComponent<React.FC<any>>;
}

const config: Array<IRoute> = [
  {
    name: 'index',
    path: '/',
    component: lazy(() => import('../pages/Index')),
  },
  {
    name: 'home',
    path: '/home',
    component: lazy(() => import('../pages/Home')),
  },
  {
    name: 'RecallFramework',
    path: '/RecallFramework',
    component: lazy(() => import('../pages/RecallFramework')),
  },
  {
    name: 'HeterogeneousPlatform',
    path: '/HeterogeneousPlatform',
    component: lazy(() => import('../pages/HeterogeneousPlatform/HeterogeneousPlatform')),
  },
  {
    name: 'Nationalkaraoke',
    path: '/HeterogeneousPlatform/Nationalkaraoke',
    component: lazy(() => import('../pages/HeterogeneousPlatform/Nationalkaraoke')),
  },
  {
    name: 'SceneRegistration',
    path: '/HeterogeneousPlatform/Nationalkaraoke/SceneRegistration',
    component: lazy(() => import('../pages/HeterogeneousPlatform/NationalkaraokeChilder/SceneRegistration/SceneRegistration')),
  },
  {
    name: 'SceneModelInformation',
    path: '/HeterogeneousPlatform/Nationalkaraoke/SceneModelInformation',
    component: lazy(() => import('../pages/HeterogeneousPlatform/NationalkaraokeChilder/SceneModelInformation/SceneModelInformation')),
  },
  {
    name: 'RegisterModelService',
    path: '/HeterogeneousPlatform/Nationalkaraoke/RegisterModelService',
    component: lazy(() => import('../pages/HeterogeneousPlatform/NationalkaraokeChilder/RegisterModelService/RegisterModelService')),
  },
  {
    name: 'RegistrationOperator',
    path: '/HeterogeneousPlatform/Nationalkaraoke/RegistrationOperator',
    component: lazy(() => import('../pages/HeterogeneousPlatform/NationalkaraokeChilder/RegistrationOperator/RegistrationOperator')),
  },
  {
    name: 'FeatureSetConfiguration',
    path: '/HeterogeneousPlatform/Nationalkaraoke/FeatureSetConfiguration',
    component: lazy(() => import('../pages/HeterogeneousPlatform/NationalkaraokeChilder/FeatureSetConfiguration/FeatureSetConfiguration')),
  },

  {
    name: 'RegisterFeaturePullService',
    path: '/HeterogeneousPlatform/Nationalkaraoke/RegisterFeaturePullService',
    component: lazy(() => import('../pages/HeterogeneousPlatform/NationalkaraokeChilder/RegisterFeaturePullService/RegisterFeaturePullService')),
  },
  {
    name: 'ReFeatureInformation',
    path: '/HeterogeneousPlatform/Nationalkaraoke/ReFeatureInformation',
    component: lazy(() => import('../pages/HeterogeneousPlatform/NationalkaraokeChilder/ReFeatureInformation/ReFeatureInformation')),
  },
  {
    name: 'FeatureConfiguration',
    path: '/HeterogeneousPlatform/Nationalkaraoke/FeatureConfiguration',
    component: lazy(() => import('../pages/HeterogeneousPlatform/NationalkaraokeChilder/FeatureConfiguration/FeatureConfiguration')),
  },
  {
    name: 'RegisterModelInformation',
    path: '/HeterogeneousPlatform/Nationalkaraoke/RegisterModelInformation',
    component: lazy(() => import('../pages/HeterogeneousPlatform/NationalkaraokeChilder/RegisterModelInformation/RegisterModelInformation')),
  },
  {
    name: 'nodeAdmin',
    path: '/HeterogeneousPlatform/componentsAdmin/nodeAdmin',
    component: lazy(() => import('../pages/HeterogeneousPlatform/componentsAdmin/nodeAdmin/index')),
  },
  {
    name: 'edgeAdmin',
    path: '/HeterogeneousPlatform/componentsAdmin/edgeAdmin',
    component: lazy(() => import('../pages/HeterogeneousPlatform/componentsAdmin/edgeAdmin/index')),
  },
  {
    name: 'structureAdmin',
    path: '/HeterogeneousPlatform/componentsAdmin/structureAdmin',
    component: lazy(() => import('../pages/HeterogeneousPlatform/componentsAdmin/structureAdmin/index')),
  },
  {
    name: 'sceneAdmin',
    path: '/HeterogeneousPlatform/componentsAdmin/sceneAdmin',
    component: lazy(() => import('../pages/HeterogeneousPlatform/componentsAdmin/sceneAdmin/index')),
  },
  {
    name: 'templateAdmin',
    path: '/HeterogeneousPlatform/componentsAdmin/templateAdmin',
    component: lazy(() => import('../pages/HeterogeneousPlatform/componentsAdmin/templateAdmin/index')),
  },
  {
    name: 'allAdmin',
    path: '/HeterogeneousPlatform/componentsAdmin/allAdmin',
    component: lazy(() => import('../pages/HeterogeneousPlatform/componentsAdmin/allAdmin/index')),
  },
  {
    name: 'ABTest',
    path: '/HeterogeneousPlatform/componentsAdmin/ABTest',
    component: lazy(() => import('../pages/HeterogeneousPlatform/componentsAdmin/ABTest/index')),
  },
  {
    name: 'chartAdmin',
    path: '/HeterogeneousPlatform/componentsAdmin/chartAdmin',
    component: lazy(() => import('../pages/HeterogeneousPlatform/componentsAdmin/chartAdmin/index')),
  },
];

export default config;

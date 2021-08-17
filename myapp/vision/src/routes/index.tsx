import React, { Suspense } from 'react';
import { Route, Switch } from 'react-router-dom';
import config, { IRoute } from './config';

const AppRouter: React.FC = () => {
  return (
    <Switch>
      {config.map((route: IRoute) => (
        <Route
          key={route.path}
          path={route.path}
          exact
          component={() => (
            <Suspense fallback={false}>
              <route.component></route.component>
            </Suspense>
          )}
        ></Route>
      ))}
    </Switch>
  );
};

export default AppRouter;

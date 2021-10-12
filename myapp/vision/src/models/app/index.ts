import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import cookie from 'cookie';
import { RootState } from '../store';

const { myapp_username, t_uid, km_uid } = cookie.parse(document.cookie);

export interface AppState {
  errMsg: any;
  userName: string;
}

const initialState: AppState = {
  errMsg: null,
  userName: myapp_username || t_uid || km_uid,
};

const AppSlice = createSlice({
  name: 'element',
  initialState,
  reducers: {
    updateErrMsg: (state, action: PayloadAction<any>) => {
      state.errMsg = action.payload;
    },
  },
});

export const { updateErrMsg } = AppSlice.actions;

export const selectErrMsg = (state: RootState): AppState['errMsg'] => state.app.errMsg;
export const selectUserName = (state: RootState): AppState['userName'] => state.app.userName;

export default AppSlice.reducer;

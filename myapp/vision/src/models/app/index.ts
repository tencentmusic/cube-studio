import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { RootState } from '../store';

export interface AppState {
  errMsg: any;
}

const initialState: AppState = {
  errMsg: null,
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

export default AppSlice.reducer;

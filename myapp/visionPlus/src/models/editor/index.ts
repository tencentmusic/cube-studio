import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { RootState } from '../store';

export interface EditorState {
  showEditor: boolean;
  key: string;
  value: string;
  type?: string
}

const initialState: EditorState = {
  showEditor: false,
  type: '',
  key: '',
  value: '',
};

const EditorSlice = createSlice({
  name: 'element',
  initialState,
  reducers: {
    updateShowEditor: (state, action: PayloadAction<boolean>) => {
      state.showEditor = action.payload;
    },
    updateKeyValue: (state, action: PayloadAction<{ key: string; value: string, type?: string }>) => {
      state.key = action.payload.key;
      state.value = action.payload.value;
      state.type = action.payload.type
    },
    updateValue: (state, action: PayloadAction<string>) => {
      state.value = action.payload;
    },
  },
});

export const { updateShowEditor, updateValue, updateKeyValue } = EditorSlice.actions;

export const selectShowEditor = (state: RootState): EditorState['showEditor'] => state.editor.showEditor;
export const selectKey = (state: RootState): EditorState['key'] => state.editor.key;
export const selectValue = (state: RootState): EditorState['value'] => state.editor.value;
export const selectType = (state: RootState): EditorState['type'] => state.editor.type;

export default EditorSlice.reducer;

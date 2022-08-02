import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { Elements } from 'react-flow-renderer';
import { RootState } from '../store';

export interface ElementState {
  elements: Elements;
  selected: Elements;
}

const initialState: ElementState = {
  elements: [],
  selected: [],
};

const elementSlice = createSlice({
  name: 'element',
  initialState,
  reducers: {
    updateElements: (state, action: PayloadAction<Elements>) => {
      state.elements = action.payload;
    },
    updateSelected: (state, action: PayloadAction<Elements>) => {
      state.selected = action.payload;
    },
  },
});

export const { updateElements, updateSelected } = elementSlice.actions;

export const selectElements = (state: RootState): ElementState['elements'] => state.element.elements;
export const selectSelected = (state: RootState): ElementState['selected'] => state.element.selected;

export default elementSlice.reducer;

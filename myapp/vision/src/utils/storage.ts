const storage = {
  get: (name: string): any => {
    const result = window.localStorage.getItem(name);

    if (result && result.match(/^(\{|\[).*(?=[}\]])/)) {
      return JSON.parse(result);
    }
    return result || '';
  },
  set: (name: string, value: unknown): void => {
    window.localStorage.setItem(name, typeof value === 'string' ? value : JSON.stringify(value));
  },
};

export default storage;

import React, { createContext, useState, useContext } from 'react';

// Create context
const AppContext = createContext();

// Context provider component
export const AppProvider = ({ children }) => {
    const [image, setImage] = useState(null);
    const [response, setResponse] = useState('');

    return (
        <AppContext.Provider value={{ image, setImage, response, setResponse }}>
            {children}
        </AppContext.Provider>
    );
};

// Custom hook to use context
export const useAppContext = () => useContext(AppContext);
